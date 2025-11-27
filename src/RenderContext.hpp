#pragma once

#include <Rose/Render/ViewportCamera.hpp>
#include "Scene/TetrahedronScene.hpp"
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Sorting/DeviceRadixSort.h>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_enums.hpp>

#define SCAN_GROUP_SIZE 256

namespace vkDelTet {

// helper for drawing with transmittance in the alpha channel
struct RenderContext {
private:
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "createPairs");
	PipelineCache updateSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "updatePairs");
	PipelineCache computeAlphaPipeline    = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));
	PipelineCache evaluateSHPipeline      = PipelineCache(FindShaderPath("EvaluateSH.cs.slang"));

	PipelineCache markPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "markTets");
	PipelineCache scanPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "prefix_sum");
	PipelineCache scatterPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "compact_tets");

	RadixSort radixSort;
	DeviceRadixSort dRadixSort;

public:
	std::optional<uint2> overrideResolution;
	TetrahedronScene    scene;
	ViewportCamera      camera;
	BufferRange<uint>   sortKeys;
	BufferRange<uint>   sortPayloads;
	BufferRange<uint2>   sortBuffer;
	BufferRange<float3> evaluatedColors;
	ImageView           renderTarget;
	BufferRange<uint>  markedTets;
	BufferRange<uint>  drawArgs;
	BufferRange<uint>  insDrawArgs;
	BufferRange<uint>  meshDrawArgs;
	BufferRange<uint>  blockSumAtomicCounter;

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

		if (!sortKeys || sortKeys.size() != scene.TetCount())
			sortKeys = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!sortBuffer || sortPayloads.size() != scene.TetCount())
			sortBuffer = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!sortPayloads || sortPayloads.size() != scene.TetCount())
			sortPayloads = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!markedTets || markedTets.size() != scene.TetCount())
			markedTets = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		uint num_groups = (scene.TetCount() + SCAN_GROUP_SIZE - 1) / SCAN_GROUP_SIZE;
		if (!blockSumAtomicCounter || blockSumAtomicCounter.size() != 1)
			blockSumAtomicCounter = Buffer::Create(context.GetDevice(), sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		if (!meshDrawArgs || meshDrawArgs.size() != 3)
			meshDrawArgs = Buffer::Create(context.GetDevice(), 4*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
		if (!drawArgs || drawArgs.size() != 4)
			drawArgs = Buffer::Create(context.GetDevice(), 4*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
		if (!insDrawArgs || insDrawArgs.size() != 5)
			insDrawArgs = Buffer::Create(context.GetDevice(), 5*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);

		ShaderParameter params = {};
		params["numSpheres"]   = scene.TetCount();
		params["sortKeys"] = (BufferParameter)sortKeys;
		params["sortPayloads"] = (BufferParameter)sortPayloads;
		createSortPairsPipeline(context, uint3(scene.TetCount(), 1u, 1u), params);
	}

	inline void PrepareRender(CommandContext& context, const float3 rayOrigin, bool prepareSH=true) {
		{
			context.PushDebugLabel("Cull");
			const uint2 extent = (uint2)renderTarget.Extent();
			const float4x4 projection = camera.GetProjection((float)extent.x / (float)extent.y);
			ShaderParameter params = {};
			params["scene"] = scene.GetShaderParameter();

			uint32_t numBlocks = (scene.TetCount() + SCAN_GROUP_SIZE - 1) / SCAN_GROUP_SIZE;
			const float4x4 cameraToWorld = camera.GetCameraToWorld();
			const float4x4 sceneToWorld  = scene.Transform();
			const float4x4 worldToScene  = inverse(sceneToWorld);
			const float4x4 sceneToCamera = inverse(cameraToWorld) * sceneToWorld;
			params["viewProjection"] = projection * sceneToCamera;
			params["invProjection"] = inverse(projection * sceneToCamera);
			params["rayOrigin"] = rayOrigin;

			params["markedTets"] = (BufferParameter)markedTets;
			params["drawArgs"] = (BufferParameter)drawArgs;
			params["insDrawArgs"] = (BufferParameter)insDrawArgs;
			params["meshDrawArgs"] = (BufferParameter)meshDrawArgs;
			params["blockSumAtomicCounter"] = (BufferParameter)blockSumAtomicCounter;
			params["numBlocks"] = numBlocks;
			params["outputResolution"] = (float2)extent;


			Pipeline& mark = *markPipeline.get(context.GetDevice());
			auto descriptorSets1 = context.GetDescriptorSets(*mark.Layout());
			context.UpdateDescriptorSets(*descriptorSets1, params, *mark.Layout());
			context.Dispatch(mark, scene.TetCount(), *descriptorSets1);
			context.Fill(blockSumAtomicCounter.cast<uint32_t>(), 0u);

			context.AddBarrier(markedTets, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();

			Pipeline& scan = *scanPipeline.get(context.GetDevice());
			auto descriptorSets2 = context.GetDescriptorSets(*scan.Layout());
			context.UpdateDescriptorSets(*descriptorSets2, params, *scan.Layout());
			context.Dispatch(scan, scene.TetCount(), *descriptorSets2);
			context.AddBarrier(blockSumAtomicCounter, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();
		}

		// Sort tetrahedra by power of circumsphere
		{
			context.PushDebugLabel("Sort");

			ShaderParameter params = {};
			params["spheres"]    = (BufferParameter)scene.TetCircumspheres();
			params["numSpheres"] = (uint32_t)scene.TetCircumspheres().size();
			params["sortKeys"] = (BufferParameter)sortKeys;
			params["sortPayloads"] = (BufferParameter)sortPayloads;
			params["rayOrigin"] = rayOrigin;
			params["markedTets"] = (BufferParameter)markedTets;

			Pipeline& updateSortPairs = *updateSortPairsPipeline.get(context.GetDevice());
			auto descriptorSets = context.GetDescriptorSets(*updateSortPairs.Layout());
			context.UpdateDescriptorSets(*descriptorSets, params, *updateSortPairs.Layout());
			context.Dispatch(updateSortPairs, scene.TetCount(), *descriptorSets);

			dRadixSort(context, sortKeys, sortPayloads);

			context.PopDebugLabel();
		}

		// evaluate tet SH coefficients
		if (prepareSH) {
			context.PushDebugLabel("EvaluateSH");

			ShaderParameter params = {};
			params["scene"]            = scene.GetShaderParameter();
			for (uint32_t i = 0; i < scene.TetSH().size(); i++)
				params["shCoeffs"][i] = (BufferParameter)scene.TetSH()[i];
			params["outputColors"]    = (BufferParameter)evaluatedColors;
			params["tetCentroids"]    = (BufferParameter)scene.TetCentroids();
			params["tetOffsets"]    = (BufferParameter)scene.TetOffsets();
			params["rayOrigin"] = rayOrigin;
			params["numPrimitives"] = scene.TetCount();
			params["markedTets"] = (BufferParameter)markedTets;

			evaluateSHPipeline(
				context,
				uint3(scene.TetCount(), 1u, 1u),
				params,
				ShaderDefines{ { "NUM_COEFFS", std::to_string(scene.NumSHCoeffs()) }}
			);

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
