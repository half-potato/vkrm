#pragma once

#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>

#include "Camera.hpp"

namespace RoseEngine {

class DelaunayTetRenderer {
private:
	// render configuration
	uint32_t renderMode = 0;
	float    pointSize = 50;
	float    densityScale = 1;
	float    percentTets = 1.f;
	float    densityThreshold = 0.f;
	bool     enableSorting = true;
	bool     reorderTets = false;

	// pipelines for rendering
	std::vector<PipelineCache> rasterPipelines;
	PipelineCache computeCircumspheresPipeline = PipelineCache(FindShaderPath("Circumsphere.cs.slang"));
	PipelineCache createSortPairsPipeline      = PipelineCache(FindShaderPath("TetSort.cs.slang"), "createPairs");
	PipelineCache reorderTetPipeline           = PipelineCache(FindShaderPath("TetSort.cs.slang"), "reorderTets");
	PipelineCache computeAlphaPipeline         = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));

	RadixSort radixSort;

	// runtime data

	Camera camera;

	BufferRange<float3> vertices;
	BufferRange<float4> vertexColors;
	BufferRange<uint4>  indices;
	BufferRange<float4> spheres;
	float3 aabbMin;
	float3 aabbMax;

	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;

	BufferRange<float4> sortedSpheres;
	BufferRange<uint2>  sortPairs;
	BufferRange<float4> sortedColors;
	BufferRange<uint4>  sortedIndices;
	ImageView renderTarget;

	Pipeline* GetRenderPipeline(CommandContext& context);
	void ComputeSpheres(CommandContext& context);

	ShaderParameter GetSceneParameter();

public:
	void LoadScene(CommandContext& context, const std::filesystem::path& p);
	void RenderProperties(CommandContext& context);
	void RenderWidget(CommandContext& context, const double dt);
	void Render(CommandContext& context);
};

}