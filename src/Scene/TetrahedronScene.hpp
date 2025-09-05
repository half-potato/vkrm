#pragma once

#include <stack>
#include <vector>
#include <filesystem>
#include <utility>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/Scene/Mesh.hpp>

namespace vkDelTet {

using namespace RoseEngine;

// A struct to consolidate all data for a single tetrahedron, simplifying the API.
struct TetrahedronAttributes {
    float  density;
    float3 gradient;
    float4 circumsphere;

    float3 centroid;
    float  offset;
    std::vector<float3> sh_coeffs; // Holds all SH coefficients for this tet
};

class TetrahedronScene {
public:
    // --- CPU-SIDE "SOURCE OF TRUTH" DATA ---
    // These are the master copies of the scene data. All modifications and
    // application logic (like selection) should operate on these vectors.
    std::vector<float3>                vertices_cpu;
    std::vector<float>                 densities_cpu;
    std::vector<float3>                gradients_cpu;
    std::vector<uint4>                 indices_cpu;


public:
    // --- PUBLIC STATE & ACCESSORS ---
    float3 sceneTranslation = float3(0);
    float3 sceneRotation    = float3(M_PI/2, 0, 0);
    
    inline const BufferRange<float4>& TetCircumspheres() const { return tetCircumspheres; }
    inline const BufferRange<float3>& TetCentroids() const { return tetCentroids; }
    inline const BufferRange<float>& TetOffsets() const { return tetOffsets; } 
    inline const auto& TetSH() const { return tetSH; }
    inline float    MaxDensity() const { return maxDensity; }
    inline float    DensityScale() const { return densityScale; } 
    inline uint32_t TetCount()    const { return (uint32_t)indices_cpu.size(); }
    inline uint32_t VertexCount() const { return (uint32_t)vertices_cpu.size(); }
    inline uint32_t NumSHCoeffs() const { return numTetSHCoeffs; } 
    inline float4x4 Transform()   const { return glm::translate(sceneTranslation) * glm::toMat4(glm::quat(sceneRotation)) * glm::scale(float3(sceneScale)); }
    
    // GPU buffer accessors for rendering
    inline const BufferRange<float3>& GetVerticesGpu() const { return vertices; }
    inline const BufferRange<uint4>&  GetIndicesGpu()  const { return tetIndices; }
    // (Add other GPU buffer accessors as needed by your renderers)


    // --- DYNAMIC MODIFICATION API ---

    // In-place (fast) updates
    void UpdateVertices(CommandContext& context, const std::vector<std::pair<uint32_t, float3>>& updates);
	void UpdateTetDensities(CommandContext& context, const std::vector<std::pair<uint32_t, float>>& updates);
    // void UpdateTetrahedronAttributes(CommandContext& context, const std::vector<std::pair<uint32_t, TetrahedronAttributes>>& updates);
    void RemoveTetrahedra(CommandContext& context, const std::vector<uint32_t>& tet_ids_to_remove);

    // Reallocation (slow) updates
    void AddVertices(CommandContext& context, const std::vector<float3>& new_vertices);
    // void AddTetrahedra(CommandContext& context, const std::vector<uint4>& new_indices, const std::vector<TetrahedronAttributes>& new_attributes);

    // --- LIFECYCLE & UTILITY ---
    void Load(CommandContext& context, const std::filesystem::path& p);
    void Save(const std::filesystem::path& p) const;
    void DrawGui(CommandContext& context);
    ShaderParameter GetShaderParameter();
	void CalculateSpheres(CommandContext& context);

private:
    // --- GPU-SIDE "RENDERING CACHE" DATA ---
    // These are considered implementation details and are managed internally.
    PipelineCache createSpheresPipeline  = PipelineCache(FindShaderPath("GenSpheres.cs.slang"));
    PipelineCache compressColorsPipeline = PipelineCache(FindShaderPath("Compression.cs.slang"));
    
    BufferRange<float3> vertices;
    BufferRange<uint4>  tetIndices;
    // Density Buffers
    TexelBufferView     tetDensities;
    BufferRange<uint16_t> tetDensities_underlyingBuffer_u16;
    BufferRange<float>  tetDensities_underlyingBuffer_f32;
    // Other Attribute Buffers
    BufferRange<float3> tetGradients;
    BufferRange<float4> tetCircumspheres;
    BufferRange<float3> tetCentroids;
    BufferRange<float>  tetOffsets;
    std::vector<BufferRange<uint32_t>> tetSH; // Striped SH data

    // --- PRIVATE STATE ---
    bool   m_densitiesAreCompressed = false;
    float  sceneScale   = 1.f;
    float  densityScale = 1.f;
    float3 minVertex, maxVertex;
    float  maxDensity   = 0.f;
    uint32_t numTetSHCoeffs = 0;

private:
    // --- PRIVATE HELPERS ---
    /**
     * @brief Generic helper to update a GPU buffer with sparse data from the CPU.
     */
    template<typename T>
    void UpdateBufferSparse(
        CommandContext& context,
        BufferRange<T>& destinationBuffer,
        const std::vector<std::pair<uint32_t, T>>& updates)
    {
        if (updates.empty()) {
            return;
        }

        const Device& device = context.GetDevice();
        const vk::DeviceSize stagingBufferSize = updates.size() * sizeof(T);

        // 1. Create a temporary, host-visible buffer to act as the staging buffer.
        // We use the Buffer::Create overload that returns a persistently mapped buffer.
        BufferRange<T> stagingBuffer = Buffer::Create(
            device,
            stagingBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        // 2. Write data to the staging buffer and record GPU-side copy commands.
        // We can do this in a single loop.
        for (size_t i = 0; i < updates.size(); ++i) {
            const auto& [elementIndex, newData] = updates[i];

            // 2a. Copy data from CPU to the mapped staging buffer.
            // The .data() method gives us the mapped pointer.
            std::memcpy(stagingBuffer.data() + i, &newData, sizeof(T));

            // 2b. Record a command to copy this single element from the staging buffer
            // to its final destination in the device-local buffer.
            context->copyBuffer(
                **stagingBuffer.mBuffer,        // Source buffer
                **destinationBuffer.mBuffer,    // Destination buffer
                vk::BufferCopy{
                    .srcOffset = stagingBuffer.mOffset + (i * sizeof(T)),
                    .dstOffset = destinationBuffer.mOffset + (elementIndex * sizeof(T)),
                    .size      = sizeof(T)
                }
            );
        }
        // The stagingBuffer is a ref<Buffer> and will be automatically cleaned up
        // when it goes out of scope.
    }
};

inline void TetrahedronScene::AddVertices(
    CommandContext& context,
    const std::vector<float3>& new_vertices)
{
	// TODO: If this becomes a bottleneck, oversize the vertex buffer and keep track of the size manually
    if (new_vertices.empty()) {
        return;
    }

    // --- 1. Update the CPU "source of truth" vector ---
    // This part is correct and remains the same.
    vertices_cpu.insert(vertices_cpu.end(), new_vertices.begin(), new_vertices.end());

    const Device& device = context.GetDevice();

    // --- 2. Re-create the entire GPU buffer from the updated CPU vector ---
    // This single pattern handles both the initial creation (from an empty state)
    // and reallocations safely and correctly.

    // Define the necessary usage flags for a versatile vertex buffer.
    const vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | // For future copies
        vk::BufferUsageFlagBits::eTransferDst;

    // Your engine's Buffer::Create that takes a vector handles the entire
    // staging buffer process (create, map, memcpy, copy command) for you.
    // We create a brand new buffer with the complete, updated data.
    vertices = Buffer::Create(device, vertices_cpu, usage);
}

inline void TetrahedronScene::UpdateVertices(CommandContext& context, const std::vector<std::pair<uint32_t, float3>>& updates) {
    for (const auto& [index, position] : updates) {
        if (index < vertices_cpu.size()) {
            vertices_cpu[index] = position;
        }
    }
    UpdateBufferSparse<float3>(context, vertices, updates);
	CalculateSpheres(context);
}

inline void TetrahedronScene::UpdateTetDensities(CommandContext& context, const std::vector<std::pair<uint32_t, float>>& updates) {
    for (const auto& [index, density] : updates) {
        if (index < densities_cpu.size()) {
            densities_cpu[index] = density;
        }
    }
    
    if (m_densitiesAreCompressed) {
        // std::vector<std::pair<uint32_t, uint16_t>> half_updates;
        // half_updates.reserve(updates.size());
        // for (const auto& [index, density_float] : updates) {
        //     half_updates.push_back({index, glm::packHalf(glm::vec1(density_float))});
        // }
        // UpdateBufferSparse<uint16_t>(context, tetDensities_underlyingBuffer_u16, half_updates);
    } else {
        UpdateBufferSparse<float>(context, tetDensities_underlyingBuffer_f32, updates);
    }
}

inline void TetrahedronScene::RemoveTetrahedra(CommandContext& context, const std::vector<uint32_t>& tet_ids_to_remove) {
    std::vector<std::pair<uint32_t, float>> densityUpdates;
    for (const uint32_t tetId : tet_ids_to_remove) {
        if (tetId < densities_cpu.size() && densities_cpu[tetId] > 0.0f) {
             densityUpdates.push_back({tetId, 0.0f});
        }
    }
    if (!densityUpdates.empty()) {
        UpdateTetDensities(context, densityUpdates);
    }
}

inline void TetrahedronScene::CalculateSpheres(CommandContext& context) {
	ShaderParameter parameters = {};
	parameters["scene"] = GetShaderParameter();
	parameters["outputSpheres"] = (BufferParameter)tetCircumspheres;
	parameters["outputCentroids"] = (BufferParameter)tetCentroids;
	parameters["outputOffsets"] = (BufferParameter)tetOffsets;
	context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)tetCircumspheres.size(), parameters);
}

// inline void TetrahedronScene::AddTetrahedra(
//     CommandContext& context,
//     const std::vector<uint4>& new_indices,
//     const std::vector<TetrahedronAttributes>& new_attributes) 
// {
//     if (new_indices.empty() || new_indices.size() != new_attributes.size()) {
//         return;
//     }
//
//     // --- 1. Update CPU "source of truth" vectors ---
//     const size_t oldTetCount = indices_cpu.size();
//     indices_cpu.insert(indices_cpu.end(), new_indices.begin(), new_indices.end());
//     attributes_cpu.insert(attributes_cpu.end(), new_attributes.begin(), new_attributes.end());
//     
//     const Device& device = context.GetDevice();
//
//     // --- 2. Reallocate all per-tetrahedron buffers ---
//
//     // Helper lambda to apply the reallocation pattern to a buffer
//     auto reallocate_buffer = [&](auto& oldBuffer, const auto& cpu_data) {
//         using T = typename decltype(oldBuffer)::value_type;
//         const vk::BufferUsageFlags usage = oldBuffer.mBuffer->Usage() | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
//         BufferRange<T> newBuffer = Buffer::Create(device, cpu_data.size() * sizeof(T), usage);
//         if (oldTetCount > 0) {
//             context.CopyBuffer(oldBuffer, newBuffer.slice(0, oldTetCount));
//         }
//         // Get just the new data from the end of the CPU vector
//         auto newDataSpan = std::span(cpu_data).subspan(oldTetCount);
//         context.UpdateBuffer(newBuffer.slice(oldTetCount), std::vector<T>(newDataSpan.begin(), newDataSpan.end()));
//         oldBuffer = newBuffer;
//     };
//     
//     // Reallocate Indices
//     reallocate_buffer(tetIndices, indices_cpu);
//
//     // Reallocate Densities (with compression logic)
//     if (m_densitiesAreCompressed) {
//         std::vector<uint16_t> densities_half_cpu;
//         densities_half_cpu.reserve(attributes_cpu.size());
//         for(const auto& attr : attributes_cpu) { densities_half_cpu.push_back(glm::packHalf1(attr.density)); }
//         reallocate_buffer(tetDensities_underlyingBuffer_u16, densities_half_cpu);
//         tetDensities = TexelBufferView::Create(device, tetDensities_underlyingBuffer_u16, vk::Format::eR16Sfloat);
//     } else {
//         std::vector<float> densities_float_cpu;
//         densities_float_cpu.reserve(attributes_cpu.size());
//         for(const auto& attr : attributes_cpu) { densities_float_cpu.push_back(attr.density); }
//         reallocate_buffer(tetDensities_underlyingBuffer_f32, densities_float_cpu);
//         tetDensities = TexelBufferView::Create(device, tetDensities_underlyingBuffer_f32, vk::Format::eR32Sfloat);
//     }
//
//     // Reallocate Other Attributes
//     std::vector<float3> gradients, centroids;
//     std::vector<float4> circumspheres;
//     std::vector<float> offsets;
//     // (Populate these vectors from attributes_cpu)
//     for(const auto& attr : attributes_cpu) {
//         gradients.push_back(attr.gradient);
//         centroids.push_back(attr.centroid);
//         circumspheres.push_back(attr.circumsphere);
//         offsets.push_back(attr.offset);
//     }
//     reallocate_buffer(tetGradients, gradients);
//     reallocate_buffer(tetCentroids, centroids);
//     reallocate_buffer(tetCircumspheres, circumspheres);
//     reallocate_buffer(tetOffsets, offsets);
// }

}
