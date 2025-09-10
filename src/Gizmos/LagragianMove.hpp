#include <vector>
#include <set>
#include <map>
#include <queue>
#include <cmath>
#include <Eigen/Sparse>
#include <Rose/Core/RoseEngine.h>

#include <unordered_map>

using namespace RoseEngine;

// Custom hasher for int3 so we can use it in std::unordered_map
struct Int3Hasher {
    std::size_t operator()(const int3& k) const {
        // A common way to hash a 3D coordinate
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
    }
};

// Assuming float3 and uint4 are defined, e.g.:
// struct float3 { float x, y, z; };
// struct uint4 { uint32_t x, y, z, w; };

// Helper for basic float3 math
inline float distanceSquared(const float3& a, const float3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// State object to hold pre-computed data between grab and update calls
struct DeformationContext {
    // Pre-computed solver with the factorized matrix for the sub-problem
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    // A map to quickly find a vertex's position in the smaller linear system
    std::map<uint32_t, uint32_t> global_to_local_idx_map;

    // The global scene.indices_cpu of vertices we are solving for (active + boundary)
    std::vector<uint32_t> problem_vertices;

    // Store original positions of boundary vertices to use as fixed constraints
    std::vector<std::pair<uint32_t, float3>> boundary_constraints;
};

void initLa(
    DeformationContext& context,
    float radius,
	vkDelTet::TetrahedronScene& scene,
    const std::vector<uint32_t>& user_handles)
{
    if (user_handles.empty()) return;
    float cell_size = radius; // Cell size should be equal to the search radius

    std::unordered_map<int3, std::vector<uint32_t>, Int3Hasher> handle_grid;
    for (const auto& handle_idx : user_handles) {
        const auto& pos = scene.vertices_cpu[handle_idx];
        int3 cell_coord = {
            static_cast<int>(floor(pos.x / cell_size)),
            static_cast<int>(floor(pos.y / cell_size)),
            static_cast<int>(floor(pos.z / cell_size))
        };
        handle_grid[cell_coord].push_back(handle_idx);
    }

    // 2. Find Active and Boundary sets using Breadth-First Search (BFS)
    std::set<uint32_t> active_set;
    std::set<uint32_t> boundary_set;
    std::queue<uint32_t> q;
    std::vector<bool> visited(scene.vertices_cpu.size(), false);
    float radius_sq = radius * radius;

    for (const auto& handle : user_handles) {
        q.push(handle);
        visited[handle] = true;
        active_set.insert(handle);
    }

    while (!q.empty()) {
        uint32_t current_v = q.front();
        q.pop();

        for (uint32_t neighbor_idx : scene.adjacency[current_v]) {
            if (visited[neighbor_idx]) continue;
            
            // OPTIMIZED: Check if neighbor is within radius using the spatial grid
            bool is_inside = false;
            const auto& neighbor_pos = scene.vertices_cpu[neighbor_idx];
            int3 center_cell = {
                static_cast<int>(floor(neighbor_pos.x / cell_size)),
                static_cast<int>(floor(neighbor_pos.y / cell_size)),
                static_cast<int>(floor(neighbor_pos.z / cell_size))
            };

            // Check the 3x3x3 cube of cells around the neighbor
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        int3 cell_to_check = {center_cell.x + x, center_cell.y + y, center_cell.z + z};
                        
                        if (handle_grid.count(cell_to_check)) {
                            for (const auto& handle_idx : handle_grid.at(cell_to_check)) {
                                if (distanceSquared(neighbor_pos, scene.vertices_cpu[handle_idx]) < radius_sq) {
                                    is_inside = true;
                                    goto found_handle; // Break out of all 4 nested loops
                                }
                            }
                        }
                    }
                }
            }
            
        found_handle:; // Label for the goto jump

            visited[neighbor_idx] = true;
            if (is_inside) {
                active_set.insert(neighbor_idx);
                q.push(neighbor_idx);
            } else {
                boundary_set.insert(neighbor_idx);
            }
        }
    }
    
    
    
    // 3. Define the sub-problem
    context.problem_vertices.assign(active_set.begin(), active_set.end());
    context.problem_vertices.insert(context.problem_vertices.end(), boundary_set.begin(), boundary_set.end());
    
    context.global_to_local_idx_map.clear();
    for (uint32_t i = 0; i < context.problem_vertices.size(); ++i) {
        context.global_to_local_idx_map[context.problem_vertices[i]] = i;
    }

    uint32_t num_problem_vertices = context.problem_vertices.size();
    std::vector<Eigen::Triplet<double>> triplets;
    context.boundary_constraints.clear();

    // 4. Build the SMALLER linear system (Matrix A)
    std::set<uint32_t> user_handle_indices;
    for(const auto& handle : user_handles) user_handle_indices.insert(handle);

    for (uint32_t local_idx = 0; local_idx < num_problem_vertices; ++local_idx) {
        uint32_t global_idx = context.problem_vertices[local_idx];

        if (user_handle_indices.count(global_idx) || boundary_set.count(global_idx)) {
            // This is a fixed constraint (either a user handle or a boundary pin)
            triplets.emplace_back(local_idx, local_idx, 1.0);
            if(boundary_set.count(global_idx)) {
                context.boundary_constraints.push_back({global_idx, scene.vertices_cpu[global_idx]});
            }
        } else {
            // This is a free vertex, apply Laplacian rule
            double degree = 0;
            for (uint32_t neighbor_global_idx : scene.adjacency[global_idx]) {
                // Only consider neighbors that are part of our sub-problem
                if (context.global_to_local_idx_map.count(neighbor_global_idx)) {
                    uint32_t neighbor_local_idx = context.global_to_local_idx_map.at(neighbor_global_idx);
                    triplets.emplace_back(local_idx, neighbor_local_idx, -1.0);
                    degree += 1.0;
                }
            }
            triplets.emplace_back(local_idx, local_idx, degree);
        }
    }

    // 5. Factorize the matrix and store it in the context for later use
    Eigen::SparseMatrix<double> A(num_problem_vertices, num_problem_vertices);
    A.setFromTriplets(triplets.begin(), triplets.end());
    context.solver.compute(A);
}

std::vector<std::pair<uint32_t, float3>> updateLa(
    const DeformationContext& context,
    const std::vector<std::pair<uint32_t, float3>>& current_user_handles)
{
    if (context.problem_vertices.empty()) return {};

    uint32_t num_problem_vertices = context.problem_vertices.size();
    Eigen::VectorXd bx = Eigen::VectorXd::Zero(num_problem_vertices);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(num_problem_vertices);
    Eigen::VectorXd bz = Eigen::VectorXd::Zero(num_problem_vertices);

    // 1. Build the right-hand-side 'b' vector with current constraint positions
    for (const auto& handle : current_user_handles) {
        // Ensure the handle is part of the problem (it always should be)
        if (context.global_to_local_idx_map.count(handle.first)) {
            uint32_t local_idx = context.global_to_local_idx_map.at(handle.first);
            bx(local_idx) = handle.second.x;
            by(local_idx) = handle.second.y;
            bz(local_idx) = handle.second.z;
        }
    }
    for (const auto& constraint : context.boundary_constraints) {
        uint32_t local_idx = context.global_to_local_idx_map.at(constraint.first);
        bx(local_idx) = constraint.second.x;
        by(local_idx) = constraint.second.y;
        bz(local_idx) = constraint.second.z;
    }

    // 2. Solve the system (this is very fast since A is already factorized)
    Eigen::VectorXd xx = context.solver.solve(bx);
    Eigen::VectorXd xy = context.solver.solve(by);
    Eigen::VectorXd xz = context.solver.solve(bz);

    // 3. Format the output to contain the new positions for all affected vertices
    std::vector<std::pair<uint32_t, float3>> smooth_updates;
    smooth_updates.reserve(context.problem_vertices.size());
    
    for (uint32_t local_idx = 0; local_idx < num_problem_vertices; ++local_idx) {
        uint32_t global_idx = context.problem_vertices[local_idx];
        smooth_updates.push_back(
            {global_idx, 
             {static_cast<float>(xx(local_idx)),
              static_cast<float>(xy(local_idx)),
              static_cast<float>(xz(local_idx))}}
        );
    }
    
    return smooth_updates;
}
