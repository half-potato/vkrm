#pragma once

#include "../RenderContext.hpp"
#include <algorithm> // For std::sort
#include <Rose/Sorting/DeviceRadixSort.h>
#include <vulkan/vulkan_core.h>
#include "LagragianMove.hpp"
#include "PBDMove.hpp"

namespace vkDelTet {

// A temporary struct to hold candidates during the sorting process
struct SelectionCandidate {
	uint32_t id;
	float depth;
};
enum class SelectionState { IDLE, GRABBING };

class VertexHighlightRenderer {
public:
	std::unordered_set<uint32_t> m_selection;
private:
	DeviceRadixSort dRadixSort;
	// We store the ID of the selected vertex. -1 means nothing is selected.
	BufferRange<uint32_t> m_selectionGpuBuffer;
	std::vector<uint32_t> m_sortedCandidates;
	int m_currentIndex = -1;
	float m_selectionRadius = 10.0f; // Default radius of 10 pixels
	BufferRange<uint>   b_numCandidates;
	BufferRange<uint>   sortBuffer;
	BufferRange<uint>   sortKeys;
	BufferRange<uint>   sortPayloads;

	// Grab state
	SelectionState m_state = SelectionState::IDLE;
	DeformationContext deform_context;
	PBDContext pbd_context;
	float3 m_grabAnchorPoint3D;
	float m_grabAnchorDepth;
	float2 m_grabMouseStart;
	std::vector<uint32_t> selected;
	std::unordered_map<uint32_t, float3> m_initialVertexPositions;

	PipelineCache m_pipeline = PipelineCache({
		{ FindShaderPath("VertexHighlight.slang"), "vsmain" },
		{ FindShaderPath("VertexHighlight.slang"), "fsmain" }
	});
	PipelineCache selectVertexPipeline = PipelineCache(FindShaderPath("VertexHighlight.slang"), "select_kernel");


	inline Pipeline& GetPipeline(CommandContext& context, RenderContext& renderContext) {
		// 1. Define the opaque blend state for a single color attachment.
		vk::PipelineColorBlendAttachmentState colorBlendAttachment {
			.blendEnable = VK_FALSE, // This is the key: disable blending.
			.srcColorBlendFactor = vk::BlendFactor::eOne,
			.dstColorBlendFactor = vk::BlendFactor::eZero,
			.colorBlendOp = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eOne,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp = vk::BlendOp::eAdd,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
		};

		// This pipeline is simple: it draws a single point and should always be on top.
		GraphicsPipelineInfo pipelineInfo {
			.vertexInputState = VertexInputDescription{},
			.inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
				.topology = vk::PrimitiveTopology::ePointList },
			.rasterizationState = vk::PipelineRasterizationStateCreateInfo{
				.cullMode = vk::CullModeFlagBits::eNone },
			.multisampleState = vk::PipelineMultisampleStateCreateInfo{},
			.depthStencilState = vk::PipelineDepthStencilStateCreateInfo{
				.depthTestEnable = false,
				.depthWriteEnable = false,
			},
			.viewports = { vk::Viewport{} },
			.scissors = { vk::Rect2D{} },

			// 2. THE FIX: Place the attachment state inside your engine's 'ColorBlendState' wrapper.
			.colorBlendState = ColorBlendState{
				.attachments = { colorBlendAttachment }
			},

			.dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
			.dynamicRenderingState = DynamicRenderingState{
				.colorFormats = { renderContext.renderTarget.GetImage()->Info().format } }
		};

		return *m_pipeline.get(context.GetDevice(), {}, pipelineInfo).get();
	}
	void ResizeGpuBufferIfNeeded(CommandContext& context, size_t requiredSize) {
		// Only resize if the buffer doesn't exist or its current capacity is too small.
		if (!m_selectionGpuBuffer || m_selectionGpuBuffer.size() < requiredSize) {
			// Grow by 1.5x to avoid reallocating on every single addition.
			// Ensure a minimum size for small selections.
			size_t newCapacity = std::max(static_cast<size_t>(requiredSize * 1.5f), (size_t)32);

			// m_selectionGpuBuffer = Buffer::Create(
			// 	context.GetDevice(),
			// 	newCapacity * sizeof(uint),
			// 	vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
			m_selectionGpuBuffer = Buffer::Create(
				context.GetDevice(),
				newCapacity * sizeof(uint32_t),
				// USAGE: Must be a storage buffer for the shader to read.
				vk::BufferUsageFlagBits::eStorageBuffer,
				// MEMORY: Must be host-visible so the CPU can write to it.
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				// ALLOCATION: Must be mapped for the .data() pointer to be valid.
				VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);
			printf("Allocated new selection buffer with capacity for %zu elements.\n", newCapacity);
		}
	}
public:
	inline void PrepareBuffers(
		CommandContext& context,
		const TetrahedronScene& scene)
	{
		if (!b_numCandidates || b_numCandidates.size() != 1)
			b_numCandidates = Buffer::Create(
				context.GetDevice(),
				sizeof(uint)*1,
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
				VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		if (!sortKeys || sortKeys.size() != scene.VertexCount())
			sortKeys = Buffer::Create(
				context.GetDevice(),
				sizeof(uint)*scene.VertexCount(),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		if (!sortBuffer || sortBuffer.size() != scene.VertexCount())
			sortBuffer = Buffer::Create(
				context.GetDevice(),
				sizeof(uint2)*scene.VertexCount(),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		if (!sortPayloads || sortPayloads.size() != scene.VertexCount())
			sortPayloads = Buffer::Create(
				context.GetDevice(),
				sizeof(uint)*scene.VertexCount(),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		ResizeGpuBufferIfNeeded(context, 32);
	}

	std::vector<uint32_t> GetSelection() {
		std::unordered_set<uint32_t> finalSelection(m_selection.begin(), m_selection.end());
		if (m_currentIndex >= 0 && !m_sortedCandidates.empty()) {
			finalSelection.insert(m_sortedCandidates[m_currentIndex]);
		}
		return std::vector<uint32_t>(finalSelection.begin(), finalSelection.end());
	}

	void ExtendSelection(CommandContext& context) {
		if (m_currentIndex >= 0 && !m_sortedCandidates.empty()) {
			m_selection.insert(m_sortedCandidates[m_currentIndex]);
			ResizeGpuBufferIfNeeded(context, m_selection.size());
		}
	}

	void ClearSelection() {
		m_selection.clear();
	}

	void MoveSelection(CommandContext& context, TetrahedronScene &scene, float3 vector) {
		std::vector<std::pair<uint32_t, float3>> updates;
		for (auto i : selected) {
			float3 old_position = scene.vertices_cpu[i];
			float3 new_position = vector + old_position;
			updates.push_back(std::make_pair((uint32_t) i, new_position));
		}
		scene.UpdateVertices(context, updates);
	}

	inline SelectionState GetState() const { return m_state; }

	void BeginGrab(TetrahedronScene& scene, const glm::mat4& viewProj, const float2& currentMousePos) {
		selected = GetSelection();
		if (selected.empty()) return;

		m_initialVertexPositions.clear();

		for (const uint32_t id : selected) {
			const float3& pos = scene.vertices_cpu[id];
			m_initialVertexPositions[id] = pos;
			m_grabAnchorPoint3D += pos;
		}
		float radius = 0.5;
		// initLa(deform_context, radius, scene, selected);
		initPBD(pbd_context, radius, scene, selected);

		m_grabMouseStart = currentMousePos;
		m_state = SelectionState::GRABBING;
	}

	void UpdateGrab(
		CommandContext& context,
		TetrahedronScene& scene,
		const float2& currentMousePos,
		const uint2& viewportSize,
		const glm::mat4& viewProj,
		const float dt)
	{
		// 1. Calculate the inverse matrix for unprojection
		const glm::mat4 invViewProj = glm::inverse(viewProj);
		glm::mat3 invRot = glm::mat3(invViewProj);


		// 7. Apply the movement to all selected vertices (from their initial positions)
		std::vector<std::pair<uint32_t, float3>> updates;
		updates.reserve(m_initialVertexPositions.size());
		for (const auto& [id, initialPos] : m_initialVertexPositions) {
			glm::vec4 clipPos = viewProj * glm::vec4(initialPos, 1.0f);
			glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;

			float sens = 1.f / max(viewportSize.x, viewportSize.y) * ndcPos.z;
			float3 mouseD = float3((currentMousePos - m_grabMouseStart) * sens, 0.f);
			glm::vec3 move_vector = invRot * mouseD;
			updates.push_back({id, initialPos + move_vector});
		}

		// 8. Send the updates to the scene
		// scene.UpdateVertices(context, updates);
		// auto fullUpdates = updateLa(deform_context, updates);
		auto fullUpdates = updatePBD(scene, pbd_context, 0.016, updates);
		scene.UpdateVertices(context, fullUpdates);
	}

	void ConfirmGrab() {
		m_initialVertexPositions.clear();
		m_state = SelectionState::IDLE;
	}

	void CancelGrab(CommandContext& context, TetrahedronScene& scene) {
		// Revert to original positions
		std::vector<std::pair<uint32_t, float3>> updates;
		for (const auto& [id, initialPos] : m_initialVertexPositions) {
			updates.push_back({id, initialPos});
		}
		scene.UpdateVertices(context, updates);
		m_initialVertexPositions.clear();
		m_state = SelectionState::IDLE;
	}

	void UpdateCandidatesGPU(
		CommandContext& context, RenderContext& renderContext,
		const float2& mousePos
	) {
		printf("%f, %f\n", mousePos.x, mousePos.y);
		const uint2 extent = (uint2)renderContext.renderTarget.Extent();
		{
			context.PushDebugLabel("Select");
			const float4x4 sceneToWorld = renderContext.scene.Transform();
			const float4x4 worldToCamera = inverse(renderContext.camera.GetCameraToWorld());
			const float4x4 projection = renderContext.camera.GetProjection((float)extent.x / (float)extent.y);

			ShaderParameter params = {};
			params["scene"] = renderContext.scene.GetShaderParameter();
			params["viewProjection"] = projection * worldToCamera * sceneToWorld;
			params["selection"] = (BufferParameter)m_selectionGpuBuffer;
			params["mousePos"] = float2(mousePos.x, mousePos.y);
			params["selectionRadius"] = m_selectionRadius;
			params["outputResolution"] = float2(extent.x, extent.y);

			params["b_numCandidates"]    = (BufferParameter)b_numCandidates;
			params["sortKeys"] = (BufferParameter)sortKeys;
			params["sortPayloads"] = (BufferParameter)sortPayloads;
			params["sortBuffer"] = (BufferParameter)sortBuffer;

			Pipeline& selectVertex = *selectVertexPipeline.get(context.GetDevice());
			auto descriptorSets = context.GetDescriptorSets(*selectVertex.Layout());
			context.UpdateDescriptorSets(*descriptorSets, params, *selectVertex.Layout());
			context.Dispatch(selectVertex, renderContext.scene.VertexCount(), *descriptorSets);

			context.AddBarrier(sortKeys, {
				.stage  = vk::PipelineStageFlagBits2::eComputeShader,
				.access = vk::AccessFlagBits2::eShaderRead
			});
			context.ExecuteBarriers();

			vkDeviceWaitIdle(**context.GetDevice());
			// dRadixSort(context, sortKeys, sortPayloads);
			// vkDeviceWaitIdle(**context.GetDevice());

			context.PopDebugLabel();
		}
		void* candidate_data_ptr = b_numCandidates.data();
		size_t num_c = 0;
		size_t vc = renderContext.scene.VertexCount();
		std::memcpy(&num_c, b_numCandidates.data(), sizeof(uint));
		std::vector<uint2> c_sortBuffer;
		std::vector<uint2> filtBuffer (vc);
		c_sortBuffer.reserve(num_c);
		std::memcpy(filtBuffer.data(), sortBuffer.data(), vc * sizeof(uint2));
		for (int i=0; i<vc; i++) {
			if (filtBuffer[i].x != 0) {
				c_sortBuffer.push_back(filtBuffer[i]);
			}
		}
		std::sort(c_sortBuffer.begin(), c_sortBuffer.end(), [](const uint2& a, const uint2& b) {
			return a.x < b.x; // Sort by payload, ascending
		});
		m_sortedCandidates.resize(c_sortBuffer.size());

		// Copy the sorted vertex IDs (y) into the final member vector.
		for (size_t i = 0; i < c_sortBuffer.size(); i++) {
			m_sortedCandidates[i] = c_sortBuffer[i].y;
		}
	}

	// Public method to be called from your main loop when a vertex is selected
	void UpdateCandidates(
		const float2& mousePos,
		const glm::mat4 viewProj,
		const uint2 viewportSize,
		const std::vector<float3>& allVertices)
	{
		std::vector<SelectionCandidate> candidates;

		for (uint32_t i = 0; i < allVertices.size(); ++i) {
			glm::vec4 clipPos = viewProj * glm::vec4(allVertices[i], 1.0f);
			if (clipPos.w <= 0.0f) continue;

			glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;
			glm::vec2 screenPos = {
				(ndcPos.x + 1.0f) * 0.5f * viewportSize.x,
				(ndcPos.y + 1.0f) * 0.5f * viewportSize.y
			};

			float dx = screenPos.x - mousePos.x;
			float dy = screenPos.y - mousePos.y;
			if ((dx * dx + dy * dy) < (m_selectionRadius * m_selectionRadius)) {
				candidates.push_back({ i, clipPos.z });
			}
		}

		std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
			return a.depth < b.depth;
		});

		m_sortedCandidates.clear();
		for (const auto& candidate : candidates) {
			m_sortedCandidates.push_back(candidate.id);
		}

		m_currentIndex = m_sortedCandidates.empty() ? -1 : 0;
	}

	void CycleSelection(int delta) {
		if (m_sortedCandidates.empty()) return;
		m_currentIndex = min(
			(size_t)max(
				int64_t(m_currentIndex + delta),
				(int64_t)0u),
			m_sortedCandidates.size()-1);
	}

	void Render(CommandContext& context, RenderContext& renderContext) {
		// Render the currently selected vertex from our internal list
		auto selection = GetSelection();
		if (selection.empty()) {
			return;
		}
		// m_selectionGpuBuffer = Buffer::Create(context.GetDevice(), selection, vk::BufferUsageFlagBits::eStorageBuffer);
		std::memcpy(m_selectionGpuBuffer.data(), std::ranges::data(selection), selection.size() * sizeof(uint));

		context.PushDebugLabel("VertexHighlightRenderer");
		Pipeline& pipeline = GetPipeline(context, renderContext);
		auto descriptorSets = context.GetDescriptorSets(*pipeline.Layout());
		const uint2 extent = (uint2)renderContext.renderTarget.Extent();

		{
			const float4x4 sceneToWorld = renderContext.scene.Transform();
			const float4x4 worldToCamera = inverse(renderContext.camera.GetCameraToWorld());
			const float4x4 projection = renderContext.camera.GetProjection((float)extent.x / (float)extent.y);

			ShaderParameter params = {};
			params["scene"] = renderContext.scene.GetShaderParameter();
			params["viewProjection"] = projection * worldToCamera * sceneToWorld;
			params["selection"] = (BufferParameter)m_selectionGpuBuffer;
			params["mousePos"] = float2(0.f, 0.f);
			params["selectionRadius"] = 0.f;

			params["b_numCandidates"]    = (BufferParameter)b_numCandidates;
			params["sortKeys"] = (BufferParameter)sortKeys;
			params["sortPayloads"] = (BufferParameter)sortPayloads;

			context.UpdateDescriptorSets(*descriptorSets, params, *pipeline.Layout());
		}

		renderContext.ContinueRendering(context);
		context->setViewport(0, vk::Viewport{ 0, 0, (float)extent.x, (float)extent.y, 0, 1 });
		context->setScissor(0, vk::Rect2D{ {0, 0}, { extent.x, extent.y } });
		context->bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);
		context.BindDescriptors(*pipeline.Layout(), *descriptorSets);
		context->draw(1, selection.size(), 0, 0);
		// context->drawIndirect(
  //               **b_numCandidates.mBuffer,
  //               b_numCandidates.mOffset,
  //               1,
  //               sizeof(vk::DrawIndirectCommand)
		// );
		context->endRendering();

		context.PopDebugLabel();
	}
};

} // namespace vkDelTet
