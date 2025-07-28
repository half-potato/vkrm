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

template<typename T>
void DebugPrintBuffer(
	const RoseEngine::ref<RoseEngine::Device>& device,
    RoseEngine::BufferRange<T>  gpuSourceBuffer,
    const std::string&          label,
    size_t                      maxElementsToPrint = 32)
{
    const size_t sizeInBytes = gpuSourceBuffer.size_bytes();
    if (sizeInBytes == 0) return;

    // 1. Create a staging buffer on the CPU
    auto stagingBuffer = RoseEngine::Buffer::Create(
        *device,
        sizeInBytes,
		vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
	);

    // 2. Create a dedicated, one-shot command buffer for this operation
    auto tempContext = RoseEngine::CommandContext::Create(device);
    tempContext->Begin();

    // 3. Record the copy command and a barrier
    // Note: Your Copy function likely adds barriers, but being explicit is good.
    tempContext->AddBarrier(gpuSourceBuffer, {
        .stage  = vk::PipelineStageFlagBits2::eTransfer,
        .access = vk::AccessFlagBits2::eTransferRead
    });
    tempContext->Copy(gpuSourceBuffer, stagingBuffer);

    // 4. Submit the work and wait for the GPU to finish
    tempContext->Submit();
    device->Wait();

    // 5. Map the staging buffer's memory and print the data
	T* data = static_cast<BufferRange<T>>(stagingBuffer).data();
    if (data)
    {
        std::cout << "--- " << label << " (" << gpuSourceBuffer.size() << " elements) ---\n";
        for (size_t i = 0; i < std::min(gpuSourceBuffer.size(), maxElementsToPrint); ++i)
        {
            std::cout << "[" << i << "]: " << data[i] << "\n";
        }
        if (gpuSourceBuffer.size() > maxElementsToPrint) {
            std::cout << "...\n";
        }
        std::cout << std::flush;
    }

    // `tempContext` and `stagingBuffer` are now destroyed safely.
}



// helper for drawing with transmittance in the alpha channel
struct RenderContext {
private:
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "createPairs");
	PipelineCache updateSortPairsPipeline = PipelineCache(FindShaderPath("TetSort.cs.slang"), "updatePairs");
	PipelineCache reorderTetsPipeline     = PipelineCache(FindShaderPath("TetSort.cs.slang"), "reorderTets");
	PipelineCache computeAlphaPipeline    = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));
	PipelineCache evaluateSHPipeline      = PipelineCache(FindShaderPath("EvaluateSH.cs.slang"));

	PipelineCache markPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "markTets");
	PipelineCache scanPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "prefix_sum");
	PipelineCache scatterPipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "compact_tets");
	PipelineCache scan2Pipeline      = PipelineCache(FindShaderPath("Culling.cs.slang"), "scan_blocks_atomic");

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
	BufferRange<uint>  scannedOffsets;
	BufferRange<uint>  markedTets;
	BufferRange<uint>  drawArgs;
	BufferRange<uint>  insDrawArgs;
	BufferRange<uint>  meshDrawArgs;
	BufferRange<uint>  kernelArgs;
	BufferRange<uint>  visibleTets;
	BufferRange<uint>  blockSums;
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
		if (!scannedOffsets || scannedOffsets.size() != scene.TetCount())
			scannedOffsets = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!visibleTets || visibleTets.size() != scene.TetCount())
			visibleTets = Buffer::Create(context.GetDevice(), scene.TetCount()*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		uint num_groups = (scene.TetCount() + SCAN_GROUP_SIZE - 1) / SCAN_GROUP_SIZE;
		if (!blockSums || blockSums.size() != num_groups)
			blockSums = Buffer::Create(context.GetDevice(), num_groups*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!blockSumAtomicCounter || blockSumAtomicCounter.size() != 1)
			blockSumAtomicCounter = Buffer::Create(context.GetDevice(), sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		if (!meshDrawArgs || visibleTets.size() != 3)
			meshDrawArgs = Buffer::Create(context.GetDevice(), 4*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
		if (!drawArgs || visibleTets.size() != 4)
			drawArgs = Buffer::Create(context.GetDevice(), 4*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
		if (!insDrawArgs || visibleTets.size() != 5)
			insDrawArgs = Buffer::Create(context.GetDevice(), 5*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
		if (!kernelArgs || visibleTets.size() != 3)
			kernelArgs = Buffer::Create(context.GetDevice(), 3*sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);

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
			params["scannedOffsets"] = (BufferParameter)scannedOffsets;
			params["drawArgs"] = (BufferParameter)drawArgs;
			params["insDrawArgs"] = (BufferParameter)insDrawArgs;
			params["meshDrawArgs"] = (BufferParameter)meshDrawArgs;
			params["kernelArgs"] = (BufferParameter)kernelArgs;
			params["visibleTets"] = (BufferParameter)visibleTets;
			params["blockSums"] = (BufferParameter)blockSums;
			params["blockSumAtomicCounter"] = (BufferParameter)blockSumAtomicCounter;
			params["numBlocks"] = numBlocks;
			params["outputResolution"] = (float2)extent;


			Pipeline& mark = *markPipeline.get(context.GetDevice());
			auto descriptorSets1 = context.GetDescriptorSets(*mark.Layout());
			context.UpdateDescriptorSets(*descriptorSets1, params, *mark.Layout());
			context.Dispatch(mark, scene.TetCount(), *descriptorSets1);

			context.AddBarrier(markedTets, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();

			Pipeline& scan = *scanPipeline.get(context.GetDevice());
			auto descriptorSets2 = context.GetDescriptorSets(*scan.Layout());
			context.UpdateDescriptorSets(*descriptorSets2, params, *scan.Layout());
			context.Dispatch(scan, scene.TetCount(), *descriptorSets2);

			context.AddBarrier(scannedOffsets, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();
			Pipeline& scan2 = *scan2Pipeline.get(context.GetDevice());
			context.Fill(blockSumAtomicCounter.cast<uint32_t>(), 0u);
			auto descriptorSets3 = context.GetDescriptorSets(*scan.Layout());
			context.UpdateDescriptorSets(*descriptorSets3, params, *scan.Layout());
			context.Dispatch(scan2, numBlocks, *descriptorSets3);

			context.AddBarrier(scannedOffsets, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();

			Pipeline& scatter = *scatterPipeline.get(context.GetDevice());
			auto descriptorSets4 = context.GetDescriptorSets(*scatter.Layout());
			context.UpdateDescriptorSets(*descriptorSets4, params, *scatter.Layout());
			context.Dispatch(scatter, scene.TetCount(), *descriptorSets4);

			context.PopDebugLabel();
		}

		// Sort tetrahedra by power of circumsphere
		{
			context.PushDebugLabel("Sort");

			ShaderParameter params = {};
			params["spheres"]    = (BufferParameter)scene.TetCircumspheres();
			params["numSpheres"] = (uint32_t)scene.TetCircumspheres().size();
			params["sortKeys"] = (BufferParameter)sortKeys;
			// params["sortBuffer"] = (BufferParameter)sortBuffer;
			params["sortPayloads"] = (BufferParameter)sortPayloads;
			params["rayOrigin"] = rayOrigin;
			params["markedTets"] = (BufferParameter)markedTets;

			Pipeline& updateSortPairs = *updateSortPairsPipeline.get(context.GetDevice());
			auto descriptorSets = context.GetDescriptorSets(*updateSortPairs.Layout());
			context.UpdateDescriptorSets(*descriptorSets, params, *updateSortPairs.Layout());
			context.Dispatch(updateSortPairs, scene.TetCount(), *descriptorSets);

			dRadixSort(context, sortKeys, sortPayloads);
			// radixSort(context, sortBuffer);

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
			// params["spheres"]    = (BufferParameter)scene.TetCircumspheres();
			params["tetCentroids"]    = (BufferParameter)scene.TetCentroids();
			params["tetOffsets"]    = (BufferParameter)scene.TetOffsets();
			params["rayOrigin"] = rayOrigin;
			params["numPrimitives"] = scene.TetCount();
			params["visibleTets"] = (BufferParameter)visibleTets;
			params["drawArgs"] = (BufferParameter)drawArgs;
			params["markedTets"] = (BufferParameter)markedTets;

			evaluateSHPipeline(context, uint3(scene.TetCount(), 1u, 1u), params, ShaderDefines{ { "NUM_COEFFS", std::to_string(scene.NumSHCoeffs()) }});
			// evaluateSHPipeline.indirect(context, kernelArgs, params, ShaderDefines{ { "NUM_COEFFS", std::to_string(scene.NumSHCoeffs()) }});

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
