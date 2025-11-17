#include <vector>
#include <set>
#include <map>
#include <queue>
#include <cmath>
#include <Eigen/Sparse>
#include <Rose/Core/RoseEngine.h>

using namespace RoseEngine;

// --- PBD Data Structures ---

// A single particle in our simulation
struct Particle {
    uint32_t global_index;     // Original index in the full mesh
    float3 position;           // Current position
    float3 predicted_position; // Position used during solver iterations
    float3 velocity;           // Current velocity
    float inverse_mass;        // 0 for fixed/kinematic particles
};

// A distance constraint (edge spring)
struct DistanceConstraint {
    uint32_t p1_local_idx, p2_local_idx;
    float rest_length;
    float alpha;
};

// A volume constraint for a tetrahedron
struct VolumeConstraint {
    uint32_t p1_local_idx, p2_local_idx, p3_local_idx, p4_local_idx;
    float rest_volume;
};

// The context now holds the entire simulation state
struct PBDContext {
    std::vector<Particle> particles;
    std::map<uint32_t, uint32_t> global_to_local_idx_map;
    std::vector<DistanceConstraint> distance_constraints;
    std::vector<VolumeConstraint> volume_constraints;

    // PBD solver parameters
    float dt = 0.016f; // Timestep, e.g., for ~60fps
    int solver_iterations = 16;
};

// Helper to calculate signed volume of a tetrahedron
float signedVolumeOfTet(const float3& p1, const float3& p2, const float3& p3, const float3& p4) {
    return dot(cross(p2 - p1, p3 - p1), p4 - p1) / 6.0f;
}

float stiffness_kernel(float x) {
    float fx = (x) * (x);
    return max(0.1f*fx + 0.01f, 1e-10f);
    // return max(1e-5f*(1.f-x), 1e-10f);
}

void initPBD(
    PBDContext& context,
    float radius,
    vkDelTet::TetrahedronScene& scene,
    const std::vector<uint32_t>& user_handles)
{
    if (user_handles.empty()) {
        // Clear context if there are no handles, nothing to simulate
        context.particles.clear();
        context.distance_constraints.clear();
        context.volume_constraints.clear();
        context.global_to_local_idx_map.clear();
        return;
    }

    // --- 1. Identify Active and Boundary Vertices ---
    std::set<uint32_t> active_set;
    std::set<uint32_t> boundary_set;
    std::queue<uint32_t> q;
    std::vector<bool> visited(scene.vertices_cpu.size(), false);
    float radius_sq = radius * radius;

    // A spatial grid to quickly check if a point is near ANY handle.
    // This is more efficient than iterating through all handles for every neighbor check.
    float cell_size = radius;
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

    // Initialize BFS with user handles
    for (const auto& handle : user_handles) {
        if (!visited[handle]) {
            q.push(handle);
            visited[handle] = true;
            // Handles themselves are not "active" in a dynamic sense, but they start the search.
        }
    }

    while (!q.empty()) {
        uint32_t current_v = q.front();
        q.pop();

        for (uint32_t neighbor_idx : scene.adjacency[current_v]) {
            if (visited[neighbor_idx]) continue;
            
            // Check if this neighbor is within the radius of ANY handle
            bool is_inside_radius = false;
            const auto& neighbor_pos = scene.vertices_cpu[neighbor_idx];
            int3 center_cell = {
                static_cast<int>(floor(neighbor_pos.x / cell_size)),
                static_cast<int>(floor(neighbor_pos.y / cell_size)),
                static_cast<int>(floor(neighbor_pos.z / cell_size))
            };

            // Check the 3x3x3 cube of cells around the neighbor's position
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        int3 cell_to_check = {center_cell.x + x, center_cell.y + y, center_cell.z + z};
                        if (handle_grid.count(cell_to_check)) {
                            for (const auto& handle_idx : handle_grid.at(cell_to_check)) {
                                if (distanceSquared(neighbor_pos, scene.vertices_cpu[handle_idx]) < radius_sq) {
                                    is_inside_radius = true;
                                    goto found_handle; // Ugly, but effective to break 4 nested loops
                                }
                            }
                        }
                    }
                }
            }
        found_handle:;

            visited[neighbor_idx] = true;
            if (is_inside_radius) {
                active_set.insert(neighbor_idx);
                q.push(neighbor_idx); // Continue searching from this vertex
            } else {
                boundary_set.insert(neighbor_idx); // Pin this vertex, but don't search further from it
            }
        }
    }

    // --- 2. Build the Localized PBD Context ---
    
    // Combine all vertices that will be part of our simulation island
    std::set<uint32_t> simulation_vertices = active_set;
    simulation_vertices.insert(boundary_set.begin(), boundary_set.end());
    simulation_vertices.insert(user_handles.begin(), user_handles.end());
    
    // Create particles ONLY for the simulation vertices
    context.particles.clear();
    context.global_to_local_idx_map.clear();
    context.particles.reserve(simulation_vertices.size());

    // Create a temporary set of handles for quick lookups
    std::set<uint32_t> handle_set(user_handles.begin(), user_handles.end());

    for (uint32_t global_idx : simulation_vertices) {
        uint32_t local_idx = context.particles.size();
        context.global_to_local_idx_map[global_idx] = local_idx;
        
        Particle p;
        p.global_index = global_idx;
        p.position = scene.vertices_cpu[global_idx];
        p.predicted_position = p.position;
        p.velocity = {0, 0, 0};
        
        // Determine inverse mass: 0 for fixed particles (handles and boundary), 1 for active particles
        bool is_handle = handle_set.count(global_idx);
        bool is_boundary = boundary_set.count(global_idx);

        if (is_handle || is_boundary) {
            p.inverse_mass = 0.0f; // Kinematic/Fixed
        } else {
            p.inverse_mass = 1.0f; // Dynamic
        }
        context.particles.push_back(p);
    }

    // --- 3. Create Constraints ONLY for the Simulation Island (EFFICIENTLY) ---
    context.distance_constraints.clear();
    context.volume_constraints.clear();
    std::set<std::pair<uint32_t, uint32_t>> existing_edges;
    std::set<uint32_t> processed_tets; // Keep track of tets we've already handled


    for (uint32_t global_idx : simulation_vertices) {
        // For each vertex in our island, check its incident tetrahedra
        if (scene.vertex_to_tets.size() <= global_idx) continue; // Safety check

        for (uint32_t tet_idx : scene.vertex_to_tets[global_idx]) {
            // If we already processed this tet, skip it
            if (processed_tets.count(tet_idx)) continue;

            const auto& tet = scene.indices_cpu[tet_idx];
            uint32_t global_indices[] = {tet.x, tet.y, tet.z, tet.w};

            // Check if ALL vertices of the tet are in our simulation island.
            bool all_vertices_in_context = 
                context.global_to_local_idx_map.count(global_indices[0]) &&
                context.global_to_local_idx_map.count(global_indices[1]) &&
                context.global_to_local_idx_map.count(global_indices[2]) &&
                context.global_to_local_idx_map.count(global_indices[3]);
            
            if (all_vertices_in_context) {
                // Add Volume Constraint
                VolumeConstraint vc;
                vc.p1_local_idx = context.global_to_local_idx_map.at(global_indices[0]);
                vc.p1_local_idx = context.global_to_local_idx_map.at(global_indices[0]);
                vc.p2_local_idx = context.global_to_local_idx_map.at(global_indices[1]);
                vc.p3_local_idx = context.global_to_local_idx_map.at(global_indices[2]);
                vc.p4_local_idx = context.global_to_local_idx_map.at(global_indices[3]);
                vc.rest_volume = signedVolumeOfTet(
                    scene.vertices_cpu[tet.x], 
                    scene.vertices_cpu[tet.y], 
                    scene.vertices_cpu[tet.z], 
                    scene.vertices_cpu[tet.w]
                );
                context.volume_constraints.push_back(vc);

                // Add 6 Distance Constraints for the tetrahedron's edges
                for (int i = 0; i < 4; ++i) {
                    for (int j = i + 1; j < 4; ++j) {
                        uint32_t u = global_indices[i];
                        uint32_t v = global_indices[j];
                        if (u > v) std::swap(u, v);
                        
                        if (existing_edges.find({u, v}) == existing_edges.end()) {
                            DistanceConstraint dc;
                            dc.p1_local_idx = context.global_to_local_idx_map.at(u);
                            dc.p2_local_idx = context.global_to_local_idx_map.at(v);
                            dc.rest_length = length(scene.vertices_cpu[u] - scene.vertices_cpu[v]);
                            float k_x = 9999;
                            for (const auto& handle_idx : user_handles) {
                                const auto& handle = scene.vertices_cpu[handle_idx];
                                float k_x_i = 0.5*(
                                    length(scene.vertices_cpu[u] - handle) +
                                    length(scene.vertices_cpu[v] - handle));
                                k_x = min(k_x, k_x_i);
                            }
                            dc.alpha = 0.01;//stiffness_kernel(k_x / radius);
                            if (k_x < 0.05) {
                                dc.alpha = 0.0001;
                            }
                            // dc.alpha = stiffness_kernel(k_x / radius);
                            context.distance_constraints.push_back(dc);
                            existing_edges.insert({u, v});
                        }
                    }
                }


                // Mark this tet as processed
                processed_tets.insert(tet_idx);
            }
        }
    }
}

// --- Constraint Solver Helpers ---

// void solveDistanceConstraint(Particle& p1, Particle& p2, float rest_length) {
//     float3 delta = p2.predicted_position - p1.predicted_position;
//     float current_length = length(delta);
//     if (current_length < 1e-6f) return;
//
//     float error = current_length - rest_length;
//     float3 correction_vector = (delta / current_length) * error;
//
//     float total_inv_mass = p1.inverse_mass + p2.inverse_mass;
//     if (total_inv_mass < 1e-6f) return;
//
//     p1.predicted_position += correction_vector * (p1.inverse_mass / total_inv_mass);
//     p2.predicted_position -= correction_vector * (p2.inverse_mass / total_inv_mass);
// }
void solveDistanceConstraint(Particle& p1, Particle& p2, float rest_length, float alpha, float dt) {
    float3 delta = p2.predicted_position - p1.predicted_position;
    float current_length = length(delta);
    if (current_length < 1e-6f) return;

    float total_inv_mass = p1.inverse_mass + p2.inverse_mass;
    if (total_inv_mass < 1e-6f) return;

    // --- XPBD Calculation ---
    // alpha_tilde is the compliance scaled by the timestep
    float alpha_tilde = alpha / (dt * dt);

    // Calculate the correction lambda
    float error = current_length - rest_length;
    float lambda = -error / (total_inv_mass + alpha_tilde);

    // Calculate the correction vector
    float3 correction_vector = (delta / current_length) * lambda;

    // Apply the correction
    p1.predicted_position -= correction_vector * p1.inverse_mass;
    p2.predicted_position += correction_vector * p2.inverse_mass;
}

void solveVolumeConstraint(Particle& p1, Particle& p2, Particle& p3, Particle& p4, float rest_volume, float dt) {
    float current_volume = signedVolumeOfTet(p1.predicted_position, p2.predicted_position, p3.predicted_position, p4.predicted_position);
    float error = current_volume - rest_volume;
    
    // --- EDITED: Consistent Gradient Calculation ---
    // These are derived directly from the definition of the volume of a tetrahedron
    // V = 1/6 * dot(cross(p2 - p1, p3 - p1), p4 - p1)
    
    const float3& pos1 = p1.predicted_position;
    const float3& pos2 = p2.predicted_position;
    const float3& pos3 = p3.predicted_position;
    const float3& pos4 = p4.predicted_position;

    float3 grad1 = cross(pos2 - pos3, pos4 - pos3) / 6.0f; // Note: This is cross(p2-p3, p4-p3), not p4-p2
    float3 grad2 = cross(pos3 - pos1, pos4 - pos1) / 6.0f;
    float3 grad3 = cross(pos4 - pos1, pos2 - pos1) / 6.0f;
    float3 grad4 = cross(pos2 - pos1, pos3 - pos1) / 6.0f;
    // --- END EDIT ---

    float sum_grad_sq = dot(grad1, grad1) * p1.inverse_mass + 
                        dot(grad2, grad2) * p2.inverse_mass + 
                        dot(grad3, grad3) * p3.inverse_mass + 
                        dot(grad4, grad4) * p4.inverse_mass;

    if (sum_grad_sq < 1e-9f) return;

    // Use the mass-weighted sum in the denominator for a more stable solve
    float lambda = -error / sum_grad_sq;

    // Apply corrections scaled by inverse mass
    p1.predicted_position += grad1 * lambda * p1.inverse_mass;
    p2.predicted_position += grad2 * lambda * p2.inverse_mass;
    p3.predicted_position += grad3 * lambda * p3.inverse_mass;
    p4.predicted_position += grad4 * lambda * p4.inverse_mass;
}

// --- Main Update Function ---
// NOTE: The context MUST be mutable, so we pass it as 'PBDContext&'
std::vector<std::pair<uint32_t, float3>> updatePBD(
    vkDelTet::TetrahedronScene& scene,
    PBDContext& context,
    const float dt,
    const std::vector<std::pair<uint32_t, float3>>& current_user_handles)
{
    // 1. Update handle positions directly
    for (const auto& handle : current_user_handles) {
        uint32_t global_idx = handle.first;
        if (context.global_to_local_idx_map.count(global_idx)) {
            uint32_t local_idx = context.global_to_local_idx_map[global_idx];
            context.particles[local_idx].position = handle.second;
        }
    }

    // 2. Predict positions for all movable particles
    for (auto& p : context.particles) {
        if (p.inverse_mass > 0.0f) {
            // Apply gravity or other forces
            // p.velocity.y -= 9.81f * context.dt;
            p.predicted_position = p.position + p.velocity * context.dt;
        } else {
            p.predicted_position = p.position; // Fixed/handle particles don't predict
        }
    }

    // 3. Main solver loop: iteratively project constraints
    for (int i = 0; i < context.solver_iterations; ++i) {
        for (const auto& c : context.distance_constraints) {
            solveDistanceConstraint(
                context.particles[c.p1_local_idx], context.particles[c.p2_local_idx], c.rest_length, c.alpha,
                dt);
        }
        // for (const auto& c : context.volume_constraints) {
        //     solveVolumeConstraint(context.particles[c.p1_local_idx], context.particles[c.p2_local_idx], 
        //                           context.particles[c.p3_local_idx], context.particles[c.p4_local_idx],
        //                           c.rest_volume,
        //                           dt);
        // }
    }

    // 4. Update final positions and velocities
    std::vector<std::pair<uint32_t, float3>> all_smooth_updates;
    all_smooth_updates.reserve(context.particles.size());

    for (auto& p : context.particles) {
        if (p.inverse_mass > 0.0f) {
            p.velocity = (p.predicted_position - p.position) / context.dt;
            p.position = p.predicted_position;
        }
        all_smooth_updates.push_back({p.global_index, p.position});
    }

    return all_smooth_updates;
}
