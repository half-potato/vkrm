#pragma once

#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Scene/Mesh.hpp>

#include "Camera.hpp"

namespace vkDelTet {

class TetrahedronScene {
private:
	PipelineCache createSpheresPipeline  = PipelineCache(FindShaderPath("GenSpheres.cs.slang"));
	PipelineCache compressColorsPipeline = PipelineCache({ FindShaderPath("CompressColors.cs.slang"), "main" });

	BufferRange<float3>   vertices;
	TexelBufferView       colors;
	TexelBufferView       densities;
	BufferRange<uint4>    indices;
	BufferRange<float4>   spheres;
	TexelBufferView       lightColors;
	TexelBufferView       lightDirections;
	
	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;
	float  densityScale = 1.f;

	float3 minVertex;
	float3 maxVertex;
	float  maxDensity = 0.f;

	uint32_t numLights = 0;

	void ComputeSpheres(CommandContext& context);
	
public:
	inline uint32_t TetCount() const { return (uint32_t)spheres.size(); }
	inline uint32_t VertexCount() const { return (uint32_t)vertices.size(); }
	inline float    MaxDensity() const { return maxDensity; }
	inline float    DensityScale() const { return densityScale; }
	inline float4x4 Transform() const { return glm::translate(sceneTranslation) * glm::toMat4(glm::quat(sceneRotation)) * glm::scale(float3(sceneScale)); }
	
	ShaderParameter GetShaderParameter();

	void Load(CommandContext& context, const std::filesystem::path& p);

	void DrawGui(CommandContext& context);
};

}