#pragma once

#include "Scene/Camera.hpp"
#include "Scene/TetrahedronScene.hpp"

namespace vkDelTet {

// helper for drawing with transmittance in the alpha channel
struct RenderContext {
private:
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "createPairs");
	PipelineCache updateSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "updatePairs");
	PipelineCache reorderTetsPipeline     = PipelineCache(FindShaderPath("TetSort.cs.slang"), "reorderTets");
	PipelineCache computeAlphaPipeline    = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));

	RadixSort radixSort;

	BufferRange<uint4>  sortedIndices;
	BufferRange<float4> sortedSpheres;
	TexelBufferView     sortedColors;
	
public:
	TetrahedronScene   scene;
	Camera             camera;
	BufferRange<uint2> sortPairs;
	ImageView          renderTarget;

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

	inline void InitializeSort(CommandContext& context, const ShaderParameter& sceneParams) {
		context.PushDebugLabel("Sort");
		if (!sortPairs || sortPairs.size() != scene.TetCount()) {
			sortPairs = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer);
		}
	
		ShaderParameter params = {};
		params["scene"]     = sceneParams;
		params["sortPairs"] = (BufferParameter)sortPairs;
	
		Pipeline& createSortPairs = *createSortPairsPipeline.get(context.GetDevice());
		auto descriptorSets = context.GetDescriptorSets(*createSortPairs.Layout());
		context.UpdateDescriptorSets(*descriptorSets, params, *createSortPairs.Layout());
		context.Dispatch(createSortPairs, scene.TetCount(), *descriptorSets);
	
		context.PopDebugLabel();
	}

	inline void SortTetrahedra(CommandContext& context, ShaderParameter& sceneParams, const float3 rayOrigin, bool reorder) {
		context.PushDebugLabel("Sort");

		ShaderParameter params = {};
		params["scene"]     = sceneParams;
		params["sortPairs"] = (BufferParameter)sortPairs;
		params["rayOrigin"] = rayOrigin;
	
		Pipeline& updateSortPairs = *updateSortPairsPipeline.get(context.GetDevice());
		auto descriptorSets = context.GetDescriptorSets(*updateSortPairs.Layout());
		context.UpdateDescriptorSets(*descriptorSets, params, *updateSortPairs.Layout());
		context.Dispatch(updateSortPairs, scene.TetCount(), *descriptorSets);
	
		radixSort(context, sortPairs);
		
		if (reorder) {
			Pipeline& reorderTets = *reorderTetsPipeline.get(context.GetDevice());
			context.Dispatch(reorderTets, scene.TetCount(), *descriptorSets);
		}
	
		context.PopDebugLabel();
	}
	
	// compute alpha = 1 - T
	inline void ConvertToAlpha(CommandContext& context) {
		const uint2 extent = (uint2)renderTarget.Extent();
		ShaderParameter params = {};
		params["image"] = ImageParameter{ .image = renderTarget, .imageLayout = vk::ImageLayout::eGeneral };
		params["dim"] = extent;
		context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), extent, params);
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
		ConvertToAlpha(context);
	}
};

}