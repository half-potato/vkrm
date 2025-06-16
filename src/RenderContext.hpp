#pragma once

#include <Rose/Render/ViewportCamera.hpp>
#include "Scene/TetrahedronScene.hpp"

namespace vkDelTet {

// helper for drawing with transmittance in the alpha channel
struct RenderContext {
private:
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "createPairs");
	PipelineCache updateSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "updatePairs");
	PipelineCache reorderTetsPipeline     = PipelineCache(FindShaderPath("TetSort.cs.slang"), "reorderTets");
	PipelineCache computeAlphaPipeline    = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));
	PipelineCache evaluateSHPipeline      = PipelineCache(FindShaderPath("EvaluateSH.cs.slang"));

	RadixSort radixSort;

public:
	TetrahedronScene    scene;
	ViewportCamera      camera;
	BufferRange<uint2>  sortPairs;
	BufferRange<float3> evaluatedColors;
	ImageView           renderTarget;

	constexpr vk::PipelineColorBlendAttachmentState GetBlendState() {
		return vk::PipelineColorBlendAttachmentState {
			.blendEnable         = true,
			.srcColorBlendFactor = vk::BlendFactor::eDstAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOne,
			.colorBlendOp        = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eDstAlpha,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp        = vk::BlendOp::eAdd,
			.colorWriteMask      = vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags} };
	}

	inline void PrepareScene(CommandContext& context, const ShaderParameter& sceneParams) {
        if (!evaluatedColors || evaluatedColors.size() != scene.TetCount())
            evaluatedColors = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(float3), vk::BufferUsageFlagBits::eStorageBuffer);
		
		if (!sortPairs || sortPairs.size() != scene.TetCount())
			sortPairs = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer);

		ShaderParameter params = {};
		params["numSpheres"]   = scene.TetCount();
		params["sortPairs"] = (BufferParameter)sortPairs;
		createSortPairsPipeline(context, uint3(scene.TetCount(), 1u, 1u), params);
	}

	inline void PrepareRender(CommandContext& context, const float3 rayOrigin) {
		// Sort tetrahedra by power of circumsphere
		{
			context.PushDebugLabel("Sort");

			ShaderParameter params = {};
			params["spheres"]    = (BufferParameter)scene.TetCircumspheres();
			params["numSpheres"] = (uint32_t)scene.TetCircumspheres().size();
			params["sortPairs"] = (BufferParameter)sortPairs;
			params["rayOrigin"] = rayOrigin;
		
			Pipeline& updateSortPairs = *updateSortPairsPipeline.get(context.GetDevice());
			auto descriptorSets = context.GetDescriptorSets(*updateSortPairs.Layout());
			context.UpdateDescriptorSets(*descriptorSets, params, *updateSortPairs.Layout());
			context.Dispatch(updateSortPairs, scene.TetCount(), *descriptorSets);
		
			radixSort(context, sortPairs);
			
			context.PopDebugLabel();
		}

		// evaluate tet SH coefficients
		{
			context.PushDebugLabel("EvaluateSH");

			ShaderParameter params = {};
			for (uint32_t i = 0; i < scene.TetSH().size(); i++)
				params["shCoeffs"][i] = (BufferParameter)scene.TetSH()[i];
			params["primPositions"]   = (BufferParameter)scene.TetCentroids();
			params["outputColors"]    = (BufferParameter)evaluatedColors;
			params["rayOrigin"] = rayOrigin;
			params["numPrimitives"] = scene.TetCount();
			evaluateSHPipeline(context, uint3(scene.TetCount(), 1u, 1u), params, ShaderDefines{ { "NUM_COEFFS", std::to_string(scene.NumSHCoeffs()) }});

			context.PopDebugLabel();
		}
	}

	inline void BeginRendering(CommandContext& context) {
        context.AddBarrier(renderTarget, Image::ResourceState{
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
            .stage  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .access =  vk::AccessFlagBits2::eColorAttachmentRead|vk::AccessFlagBits2::eColorAttachmentWrite,
            .queueFamily = context.QueueFamily() });
        context.ExecuteBarriers();
            
		vk::RenderingAttachmentInfo attachments[1] = {
			vk::RenderingAttachmentInfo {
				.imageView = *renderTarget,
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.resolveMode = vk::ResolveModeFlagBits::eNone,
				.resolveImageView = {},
				.resolveImageLayout = vk::ImageLayout::eUndefined,
				.loadOp  = vk::AttachmentLoadOp::eClear,
				.storeOp = vk::AttachmentStoreOp::eStore,
				.clearValue = vk::ClearValue{vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 1 }} } } };
		context->beginRendering(vk::RenderingInfo {
			.renderArea = vk::Rect2D{ {0, 0},  { renderTarget.Extent().x, renderTarget.Extent().y } },
			.layerCount = 1,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = attachments });
	}
	
	inline void EndRendering(CommandContext& context) {
		context->endRendering();

		// compute alpha = 1 - T
		{
			const uint2 extent = (uint2)renderTarget.Extent();
			ShaderParameter params = {};
			params["image"] = ImageParameter{ .image = renderTarget, .imageLayout = vk::ImageLayout::eGeneral };
			params["dim"] = extent;
			context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), extent, params);
		}
	}
};

}