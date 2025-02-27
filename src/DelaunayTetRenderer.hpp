#pragma once

#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Scene/Mesh.hpp>

#include "Camera.hpp"

namespace RoseEngine {

class DelaunayTetRenderer {
private:
	// render configuration
	uint32_t renderMode = 0;
	bool     wireframe = false;
	float    pointSize = 20;
	float    densityScale = 1;
	float    percentTets = 1.f; // percent of tets to draw
	float    densityThreshold = 0.f;

	bool     enableSorting = true;
	bool     reorderTets = false; // reorder tet indices/colors after sorting

	bool     reorderOnLoad = false; // reorder vertices after loading a scene

	// pipelines for rendering
	std::vector<PipelineCache> rasterPipelines;
	PipelineCache createSpheresPipeline    = PipelineCache(FindShaderPath("GenSpheres.cs.slang"),   "main");
	PipelineCache createSortPairsPipeline  = PipelineCache(FindShaderPath("TetSort.cs.slang"),      "createPairs");
	PipelineCache reorderTetPipeline       = PipelineCache(FindShaderPath("TetSort.cs.slang"),      "reorderTets");
	PipelineCache computeAlphaPipeline     = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"),  "main");

	// runtime data

	RadixSort radixSort;

	Camera camera;

	BufferRange<float3> vertices;
	BufferRange<float4> colors;
	BufferRange<uint4>  indices;
	BufferRange<float4> spheres;
	
	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;

	BufferRange<float4> sortedSpheres;
	BufferRange<uint2>  sortPairs;
	BufferRange<float4> sortedColors;
	BufferRange<uint4>  sortedIndices;
	ImageView renderTarget;
	ImageView depthBuffer;

	float3 minVertex;
	float3 maxVertex;

	Pipeline* GetRenderPipeline(CommandContext& context);

	void ComputeSpheres(CommandContext& context, const ShaderParameter& scene);
	void ComputeTriangles(CommandContext& context, const ShaderParameter& scene, const float3 rayOrigin);
	void Sort(CommandContext& context, const ShaderParameter& scene, const float3 cameraPosition);

	ShaderParameter GetSceneParameter();

public:
	void LoadScene(CommandContext& context, const std::filesystem::path& p);
	void RenderProperties(CommandContext& context);
	void RenderWidget(CommandContext& context, const double dt);
	void Render(CommandContext& context);
};

}