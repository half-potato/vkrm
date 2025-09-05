#pragma once

#include "../RenderContext.hpp"
#include <algorithm> // For std::sort

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
    // We store the ID of the selected vertex. -1 means nothing is selected.
	BufferRange<uint32_t> m_selectionGpuBuffer;
    std::vector<uint32_t> m_sortedCandidates;
    int m_currentIndex = -1;
    float m_selectionRadius = 10.0f; // Default radius of 10 pixels
	BufferRange<uint>   candidateCount;
	BufferRange<uint>   sortKeys;
	BufferRange<uint>   sortPayloads;

	// Grab state
	SelectionState m_state = SelectionState::IDLE;
    float3 m_grabAnchorPoint3D;
    float m_grabAnchorDepth;
    std::unordered_map<uint32_t, float3> m_initialVertexPositions;

    PipelineCache m_pipeline = PipelineCache({
        { FindShaderPath("VertexHighlight.slang"), "vsmain" },
        { FindShaderPath("VertexHighlight.slang"), "fsmain" }
    });
	PipelineCache select_pipeline = PipelineCache(FindShaderPath("VertexHighlight.slang"), "select");


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
		if (!candidateCount || candidateCount.size() != 1)
			candidateCount = Buffer::Create(context.GetDevice(), sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		if (!sortKeys || sortKeys.size() != scene.VertexCount())
			sortKeys = Buffer::Create(context.GetDevice(), sizeof(uint)*scene.VertexCount(), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		if (!sortPayloads || sortPayloads.size() != scene.VertexCount())
			sortPayloads = Buffer::Create(context.GetDevice(), sizeof(float)*scene.VertexCount(), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
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
		auto selection = GetSelection();
		for (auto i : selection) {
			float3 old_position = scene.vertices_cpu[i];
			float3 new_position = vector + old_position;
			updates.push_back(std::make_pair((uint32_t) i, new_position));
		}
		scene.UpdateVertices(context, updates);
	}

    inline SelectionState GetState() const { return m_state; }

	void BeginGrab(TetrahedronScene& scene, const glm::mat4& viewProj) {
		auto selection = GetSelection();
		if (selection.empty()) return;
		printf("Begin grab\n");

		m_state = SelectionState::GRABBING;
		m_initialVertexPositions.clear();
		
		// Calculate the average position (anchor) and store initial positions
		m_grabAnchorPoint3D = {0,0,0};
		for (const uint32_t id : selection) {
			const float3& pos = scene.vertices_cpu[id];
			m_initialVertexPositions[id] = pos;
			m_grabAnchorPoint3D += pos;
		}
		m_grabAnchorPoint3D /= selection.size();

		// Project the 3D anchor to get the depth in NDC [-1, 1] for the move plane
		glm::vec4 clipPos = viewProj * glm::vec4(m_grabAnchorPoint3D, 1.0f);
		if (clipPos.w != 0.0f) {
			m_grabAnchorDepth = clipPos.z / clipPos.w; // Corrected depth
		} else {
			m_grabAnchorDepth = 0.0f;
		}
	}

	void UpdateGrab(
		CommandContext& context,
		TetrahedronScene& scene,
		const ImVec2& currentMousePos,
		const uint2& viewportSize,
		const glm::mat4& viewProj)
	{
		printf("Update grab\n");
		// 1. Calculate the inverse matrix for unprojection
		const glm::mat4 invViewProj = glm::inverse(viewProj);

		// 2. Convert mouse coordinates from screen space [0, width] to NDC [-1, 1]
		float mouseX_ndc = (currentMousePos.x / viewportSize.x) * 2.0f - 1.0f;
		float mouseY_ndc = (currentMousePos.y / viewportSize.y) * 2.0f - 1.0f;
		
		// In NDC, Y is often flipped compared to screen coordinates.
		// If your movement feels inverted on the Y axis, uncomment the next line.
		// mouseY_ndc = -mouseY_ndc;

		// 3. Create a 4D point in NDC space using the stored depth
		glm::vec4 newNdcPoint(mouseX_ndc, mouseY_ndc, m_grabAnchorDepth, 1.0f);

		// 4. Unproject the point from NDC back to world space
		glm::vec4 newWorldPoint = invViewProj * newNdcPoint;

		// 5. Perform perspective divide to get the final 3D coordinates
		if (newWorldPoint.w != 0.0f) {
			newWorldPoint /= newWorldPoint.w;
		}

		// 6. Calculate the 3D movement vector
		glm::vec3 move_vector = glm::vec3(newWorldPoint) - m_grabAnchorPoint3D;

		// 7. Apply the movement to all selected vertices (from their initial positions)
		std::vector<std::pair<uint32_t, float3>> updates;
		updates.reserve(m_initialVertexPositions.size());
		for (const auto& [id, initialPos] : m_initialVertexPositions) {
			updates.push_back({id, initialPos + move_vector});
		}
		
		// 8. Send the updates to the scene
		scene.UpdateVertices(context, updates);
	}
    
    void ConfirmGrab() {
		printf("Confirm grab\n");
        m_state = SelectionState::IDLE;
        m_initialVertexPositions.clear();
    }

    void CancelGrab(CommandContext& context, TetrahedronScene& scene) {
		printf("Cancel grab\n");
        m_state = SelectionState::IDLE;
        
        // Revert to original positions
        std::vector<std::pair<uint32_t, float3>> updates;
        for (const auto& [id, initialPos] : m_initialVertexPositions) {
            updates.push_back({id, initialPos});
        }
        scene.UpdateVertices(context, updates);
        m_initialVertexPositions.clear();
    }

    // Public method to be called from your main loop when a vertex is selected
    void UpdateCandidates(
        const ImVec2& mousePos,
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

            context.UpdateDescriptorSets(*descriptorSets, params, *pipeline.Layout());
        }

        renderContext.ContinueRendering(context);
        context->setViewport(0, vk::Viewport{ 0, 0, (float)extent.x, (float)extent.y, 0, 1 });
        context->setScissor(0, vk::Rect2D{ {0, 0}, { extent.x, extent.y } });
        context->bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);
        context.BindDescriptors(*pipeline.Layout(), *descriptorSets);
        context->draw(1, selection.size(), 0, 0);
		context->endRendering();

        context.PopDebugLabel();
    }
};

} // namespace vkDelTet
