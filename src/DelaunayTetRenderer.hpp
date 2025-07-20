#pragma once

#include <stack>

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/RadixSort/RadixSort.hpp>
#include <Rose/Scene/Mesh.hpp>

// #include "Renderers/MeshShaderRenderer.hpp"
#include "Renderers/RasterRenderer.hpp"
#include "Renderers/PointCloudRenderer.hpp"
#include "Renderers/BillboardRenderer.hpp"

namespace vkDelTet {

class DelaunayTetRenderer {
private:
	std::tuple<
		// MeshShaderRenderer,
		// BillboardRenderer,
		RasterRenderer,
		PointCloudRenderer
	> renderers;
	uint32_t rendererIndex = 0;

	RenderContext renderContext;

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
	inline void LoadScene(CommandContext& context, const std::filesystem::path& p) {
		renderContext.scene.Load(context, p);
		if (renderContext.scene.VertexCount() > 0)
			renderContext.PrepareScene(context, renderContext.scene.GetShaderParameter());
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
		const float2 extentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
		const uint2 extent = uint2(extentf);
		if (extent.x == 0 || extent.y == 0) return;

		if (!renderContext.renderTarget || renderContext.renderTarget.Extent().x != extent.x || renderContext.renderTarget.Extent().y != extent.y) {
			renderContext.renderTarget = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR8G8B8A8Unorm,
					.extent = uint3(extent, 1),
					.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
					.queueFamilies = { context.QueueFamily() } }),
				vk::ImageSubresourceRange{
					.aspectMask = vk::ImageAspectFlagBits::eColor,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 });
		}

		// Draw the renderTarget image to the window
		ImGui::Image(Gui::GetTextureID(renderContext.renderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(extentf));

		renderContext.camera.Update(dt);

		// render the scene into renderTarget
		
		if (renderContext.scene.TetCount() == 0) {
			context.ClearColor(renderContext.renderTarget, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 0 }});
		} else {
			context.PushDebugLabel("DelaunayTetRenderer::Render");
			CallRendererFn([&](auto& r){ r.Render(context, renderContext); });
			context.PopDebugLabel();
		}
	}
};

}
