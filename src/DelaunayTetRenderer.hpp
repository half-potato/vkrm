#pragma once

#include <iterator>
#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Scene/Mesh.hpp>

#include "Renderers/MeshShaderRenderer.hpp"
#include "Renderers/RasterRenderer.hpp"
#include "Renderers/InstancedRenderer.hpp"
#include "Renderers/PointCloudRenderer.hpp"
#include "Renderers/BillboardRenderer.hpp"
#include "Gizmos/VertexSelection.hpp"

namespace vkDelTet {

class DelaunayTetRenderer {
private:
	std::tuple<
		MeshShaderRenderer,
		InstancedRenderer,
		BillboardRenderer,
		RasterRenderer,
		PointCloudRenderer
	> renderers;
	VertexHighlightRenderer m_highlightRenderer;

	uint32_t rendererIndex = 0;


	template<size_t I>
	inline auto CallRendererFn_(auto&& fn, uint32_t idx) {
		if (idx == I) {
			return fn( std::get<I>(renderers) );
		} else if constexpr (I+1 < std::tuple_size_v<decltype(renderers)>) {
			return CallRendererFn_<I + 1>(fn, idx);
		}
		std::unreachable();
	}
	inline auto CallRendererFn(auto&& fn, uint32_t idx) { return CallRendererFn_<0>(fn, idx); }
	inline auto CallRendererFn(auto&& fn) { return CallRendererFn_<0>(fn, rendererIndex); }

public:
	uint64_t frame_count = 0;
	RenderContext renderContext;
	inline void LoadScene(CommandContext& context, const std::filesystem::path& p) {
		renderContext.scene.Load(context, p);
		if (renderContext.scene.VertexCount() > 0) {
			renderContext.PrepareScene(context, renderContext.scene.GetShaderParameter());
			m_highlightRenderer.PrepareBuffers(context, renderContext.scene);
		}
	}

	inline void DrawPropertiesGui(CommandContext& context) {
		if (ImGui::CollapsingHeader("Camera")) {
			renderContext.camera.DrawInspectorGui();
		}

		if (ImGui::CollapsingHeader("Scene")) {
			renderContext.scene.DrawGui(context);
		}

		if (ImGui::CollapsingHeader("Renderer")) {
			if (renderContext.renderTarget) {
				ImGui::Text("%u x %u", renderContext.renderTarget.Extent().x, renderContext.renderTarget.Extent().y);
			}

			if (ImGui::BeginCombo("Mode", CallRendererFn([](const auto& r) { return r.Name(); }))) {
				auto drawComboItem = [&](const auto& r, uint32_t i) {
					if (ImGui::Selectable(r.Name(), rendererIndex == i)) {
						rendererIndex = i;
					}
				};
				// https://stackoverflow.com/questions/78863041/getting-index-of-current-tuple-item-in-stdapply
				[&]<std::size_t... Is>(std::index_sequence<Is...>) {
					((drawComboItem(std::get<Is>(renderers), (uint32_t)Is)), ...);
				}
				( std::make_index_sequence<std::tuple_size_v<decltype(renderers)>>{} );
				
				ImGui::EndCombo();
			}

			CallRendererFn([&](auto& r){ r.DrawGui(context); });
		}
	}

	void DrawWidgetGui(CommandContext& context, const double dt) {
		// Get the size of the ImGui viewport for display
		const float2 displayExtentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());

		// Determine the rendering resolution
		uint2 renderExtent;
		if (renderContext.overrideResolution) {
			// Use the override resolution if it exists
			renderExtent = *renderContext.overrideResolution;
		} else {
			// Otherwise, use the display viewport size
			renderExtent = uint2(displayExtentf);
		}

		if (renderExtent.x == 0 || renderExtent.y == 0) return;

		// Resize the render target only when the *render* extent changes
		if (!renderContext.renderTarget || renderContext.renderTarget.Extent().x != renderExtent.x || renderContext.renderTarget.Extent().y != renderExtent.y) {
			renderContext.renderTarget = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR8G8B8A8Unorm,
					.extent = uint3(renderExtent, 1), // Use renderExtent here
					.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
					.queueFamilies = { context.QueueFamily() } }),
				vk::ImageSubresourceRange{
					.aspectMask = vk::ImageAspectFlagBits::eColor,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 });
		}

		// Draw the renderTarget image, scaling it to the ImGui window size
		ImGui::Image(Gui::GetTextureID(renderContext.renderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(displayExtentf));

        const uint2    extent = (uint2)renderContext.renderTarget.Extent();

        const float4x4 sceneToWorld  = renderContext.scene.Transform();
		const float4x4 worldToCamera = inverse(renderContext.camera.GetCameraToWorld());
		const float4x4 projection = renderContext.camera.GetProjection((float)extent.x / (float)extent.y);
		const float4x4 viewmat = projection * worldToCamera * sceneToWorld;
		renderContext.camera.Update(dt);
		if (ImGui::IsWindowHovered()) {
			ImVec2 absoluteMousePos = ImGui::GetMousePos();
			ImVec2 windowPos = ImGui::GetWindowPos();
			ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
			ImVec2 viewportMin = {windowPos.x + contentMin.x, windowPos.y + contentMin.y};
			float2 relativeMousePos = {absoluteMousePos.x - viewportMin.x, absoluteMousePos.y - viewportMin.y};
			if (m_highlightRenderer.GetState() == SelectionState::IDLE) {
				// SELECTING NEW POINTS
				if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
					if (ImGui::IsKeyDown(ImGuiMod_Shift)) {
						m_highlightRenderer.ExtendSelection(context);
					} else {
						m_highlightRenderer.ClearSelection();
					}

					m_highlightRenderer.UpdateCandidates(
						relativeMousePos, viewmat, extent, 
						renderContext.scene.vertices_cpu);
					// m_highlightRenderer.UpdateCandidatesGPU(context, renderContext, relativeMousePos);
				} else {
					// ADJUSTING SELECTION
					float mouseWheel = ImGui::GetIO().MouseWheel;
					if (mouseWheel > 0) {
						m_highlightRenderer.CycleSelection(1);
					} else if (mouseWheel < 0) {
						m_highlightRenderer.CycleSelection(-1);
					} else {
						// MOVING POINTS
						if (ImGui::IsKeyPressed(ImGuiKey_G)) {
							m_highlightRenderer.BeginGrab(renderContext.scene, viewmat, relativeMousePos);
						}
					}
				}
			} else if (m_highlightRenderer.GetState() == SelectionState::GRABBING) {

				if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
					m_highlightRenderer.ConfirmGrab();
				} else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
					m_highlightRenderer.CancelGrab(context, renderContext.scene);
				} else {
					m_highlightRenderer.UpdateGrab(context, renderContext.scene, relativeMousePos, extent, viewmat, dt);
				}
			}
		}

		if (renderContext.scene.TetCount() == 0) {
			context.ClearColor(renderContext.renderTarget, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 0 }});
		} else {
			context.PushDebugLabel("DelaunayTetRenderer::Render");
			CallRendererFn([&](auto& r){ r.Render(context, renderContext); });
			context.PopDebugLabel();
		}


		m_highlightRenderer.Render(context, renderContext);

		frame_count++;
	}
};

}
