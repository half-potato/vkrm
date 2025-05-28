#pragma once

#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Scene/Mesh.hpp>

namespace vkDelTet {

using namespace RoseEngine;

class TetrahedronScene {
private:
	PipelineCache createSpheresPipeline  = PipelineCache(FindShaderPath("GenSpheres.cs.slang"));
	PipelineCache compressColorsPipeline = PipelineCache(FindShaderPath("Compression.cs.slang"));

	BufferRange<float3>   vertices;
	TexelBufferView       vertexSH;
	BufferRange<uint4>    tetIndices;
	TexelBufferView       tetDensities;
	BufferRange<float3>   tetGradients;
	BufferRange<float4>   tetCircumspheres;
	
	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;
	float  densityScale = 1.f;

	float3 minVertex;
	float3 maxVertex;
	float  maxDensity = 0.f;

	void ComputeSpheres(CommandContext& context);
	
public:
	inline uint32_t TetCount() const { return (uint32_t)tetIndices.size(); }
	inline uint32_t VertexCount() const { return (uint32_t)vertices.size(); }
	inline float    MaxDensity() const { return maxDensity; }
	inline float    DensityScale() const { return densityScale; }
	inline float4x4 Transform() const { return glm::translate(sceneTranslation) * glm::toMat4(glm::quat(sceneRotation)) * glm::scale(float3(sceneScale)); }
	
	ShaderParameter GetShaderParameter();

	void Load(CommandContext& context, const std::filesystem::path& p);

	void DrawGui(CommandContext& context);
};

}