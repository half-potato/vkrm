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
	BufferRange<float3>   tetCentroids;
	BufferRange<uint32_t> tetSH; // fp16 or fp32
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
	uint32_t numTetSHCoeffs = 0; // does not count the 0th
	float  maxDensity = 0.f;
	
public:
	inline uint32_t TetCount() const { return (uint32_t)tetIndices.size(); }
	inline uint32_t VertexCount() const { return (uint32_t)vertices.size(); }
	inline uint32_t NumSHCoeffs() const { return numTetSHCoeffs; }
	inline const BufferRange<float4>& TetCircumspheres() const { return tetCircumspheres; }
	inline const BufferRange<float3>& TetCentroids() const { return tetCentroids; }
	inline const BufferRange<uint32_t>& TetSH() const { return tetSH; }
	inline float    MaxDensity() const { return maxDensity; }
	inline float    DensityScale() const { return densityScale; }
	inline float4x4 Transform() const { return glm::translate(sceneTranslation) * glm::toMat4(glm::quat(sceneRotation)) * glm::scale(float3(sceneScale)); }
	
	ShaderParameter GetShaderParameter();

	void DrawGui(CommandContext& context);

	void Load(CommandContext& context, const std::filesystem::path& p);
};

}