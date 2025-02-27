#include <portable-file-dialogs.h>
#include <tinyply.h>

#include <Rose/Scene/Mesh.hpp>

#include "DelaunayTetRenderer.hpp"
#include "SortUtils.h"

using namespace RoseEngine;

// Configurations
namespace {
struct RasterConfig {
	const char* name;
	const char* vs = nullptr;
	const char* fs = nullptr;
	ShaderDefines defs = {};
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
	vk::PolygonMode polygonMode = vk::PolygonMode::eFill;
};
static const RasterConfig kRasterConfigurations[] {
	{
		.name = "Full tets (frag intersection)",
		.vs = "vs_tet",
		.fs = "fs_isect",
		.defs = {},
		.topology = vk::PrimitiveTopology::eTriangleList,
		.polygonMode = vk::PolygonMode::eFill,
	},
	{
		.name = "Full tets (t buffer)",
		.vs = "vs_tet",
		.fs = "fs_depth",
		.defs = { { "VS_NO_ISECT", "1" } },
		.topology = vk::PrimitiveTopology::eTriangleList,
		.polygonMode = vk::PolygonMode::eFill,
	},
	{
		.name = "Full tets (constant density)",
		.vs = "vs_tet",
		.fs = "fs_color",
		.defs = { { "VS_NO_ISECT", "1" }, { "SCALE_DENSITY_BY_SIZE", "1" } },
		.topology = vk::PrimitiveTopology::eTriangleList,
		.polygonMode = vk::PolygonMode::eFill,
	},
	{
		.name = "Billboards",
		.vs = "vs_sphere_center",
		.fs = "fs_isect",
		.defs = {},
		.topology = vk::PrimitiveTopology::ePointList,
		.polygonMode = vk::PolygonMode::ePoint,
	},
	{
		.name = "Circumcenters",
		.vs = "vs_sphere_center",
		.fs = "fs_color",
		.defs = { { "VS_NO_ISECT", "1" }, { "SCALE_DENSITY_BY_SIZE", "1" } },
		.topology = vk::PrimitiveTopology::ePointList,
		.polygonMode = vk::PolygonMode::ePoint,
	},
};
}

void OptimizeScene(std::span<const float3> vertices, std::span<const uint4> indices, std::span<const float4> colors) {
	
}

void DelaunayTetRenderer::LoadScene(CommandContext& context, const std::filesystem::path& p) {
	if (!std::filesystem::exists(p))
		return;

	std::ifstream file;
	file.open(p, std::ios::binary);

	tinyply::PlyFile ply;
	ply.parse_header(file);

	auto ply_vertex_xyz  = ply.request_properties_from_element("vertex", { "x", "y", "z" });
	auto ply_vertex_col  = ply.request_properties_from_element("tetrahedron", { "r", "g", "b", "s" });
	auto ply_tetrahedron = ply.request_properties_from_element("tetrahedron", { "vertex_indices" }, 4);

	ply.read(file);

	std::span pos {reinterpret_cast<float3*>(ply_vertex_xyz ->buffer.get()), ply_vertex_xyz ->buffer.size_bytes()/sizeof(float3)};
	std::span col {reinterpret_cast<float4*>(ply_vertex_col ->buffer.get()), ply_vertex_col ->buffer.size_bytes()/sizeof(float4)};
	std::span inds{reinterpret_cast<uint4* >(ply_tetrahedron->buffer.get()), ply_tetrahedron->buffer.size_bytes()/sizeof(uint4)};

	uint32_t numTets = (uint32_t)inds.size();

	//std::cout << "First tet:" << inds[0] << std::endl;
	//std::cout << "\tpos " << pos[inds[0][0]] << "   " << pos[inds[0][1]] << "   " << pos[inds[0][2]] << "   " << pos[inds[0][3]] << "   " << std::endl;
	//std::cout << "\tcol " << col[0] << std::endl;

	if (reorderOnLoad) {
		OptimizeScene(pos, inds, col);
	}

	// fix winding order
	{
		uint c = 0;
		for (uint4& tet : inds) {
			float3 v0 = pos[tet[0]];
			float3 v1 = pos[tet[1]];
			float3 v2 = pos[tet[2]];
			float3 v3 = pos[tet[3]];

			float3 u = v1 - v0;
			float3 v = v2 - v0;
			float3 w = v3 - v0;

			if (dot(cross(u,v), w) > 0) {
				c++;
			}
		}
		std::cout << c << " (" << ((100*c) / inds.size()) <<"%)" << std::endl;
	}

	context.GetDevice().Wait(); // wait in case previous vertices/colors/indices are in use still

	vertices = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eVertexBuffer);
	colors   = context.UploadData(col,  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eVertexBuffer);
	indices  = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	spheres            = Buffer::Create(context.GetDevice(), numTets*sizeof(float4),        vk::BufferUsageFlagBits::eStorageBuffer);

	minVertex = float3( FLT_MAX );
	maxVertex = float3( FLT_MIN );
	for (float3 p : pos) {
		minVertex = min(p, minVertex);
		maxVertex = max(p, maxVertex);
	}

	ComputeSpheres(context, GetSceneParameter());
}

Pipeline* DelaunayTetRenderer::GetRenderPipeline(CommandContext& context) {
	if (rasterPipelines.size() < std::ranges::size(kRasterConfigurations))
		rasterPipelines.resize(std::ranges::size(kRasterConfigurations));
		
	const RasterConfig& cfg = kRasterConfigurations[renderMode];

	auto& pipeline = rasterPipelines[renderMode];
	if (!pipeline) {
		pipeline = PipelineCache({
			{ FindShaderPath("DelaunayTetRenderer.3d.slang"), cfg.vs },
			{ FindShaderPath("DelaunayTetRenderer.3d.slang"), cfg.fs }
		});
	}

	ShaderDefines defines = cfg.defs;
	defines["REORDER_TETS"] = reorderTets ? "1" : "0";

	GraphicsPipelineInfo pipelineInfo {
		.vertexInputState = VertexInputDescription{},
		.inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
			.topology = cfg.topology },
		.rasterizationState = vk::PipelineRasterizationStateCreateInfo{
			.depthClampEnable = false,
			.rasterizerDiscardEnable = false,
			.polygonMode = (wireframe && cfg.polygonMode == vk::PolygonMode::eFill ? vk::PolygonMode::eLine : cfg.polygonMode),
			.cullMode = vk::CullModeFlagBits::eFront,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = false },
		.multisampleState = vk::PipelineMultisampleStateCreateInfo{},
		.depthStencilState = vk::PipelineDepthStencilStateCreateInfo{
			.depthTestEnable = false,
			.depthWriteEnable = false,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = false,
			.stencilTestEnable = false },
		.viewports = { vk::Viewport{} },
		.scissors = { vk::Rect2D{} },
		.colorBlendState = ColorBlendState{
			.attachments = { vk::PipelineColorBlendAttachmentState {
				.blendEnable         = true,
				.srcColorBlendFactor = vk::BlendFactor::eDstAlpha,
				.dstColorBlendFactor = vk::BlendFactor::eOne,
				.colorBlendOp        = vk::BlendOp::eAdd,
				.srcAlphaBlendFactor = vk::BlendFactor::eDstAlpha,
				.dstAlphaBlendFactor = vk::BlendFactor::eZero,
				.alphaBlendOp        = vk::BlendOp::eAdd,
				.colorWriteMask      = vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags} } } },
		.dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
		.dynamicRenderingState = DynamicRenderingState{
			.colorFormats = { renderTarget.GetImage()->Info().format } } };

	return pipeline.get(context.GetDevice(), defines, pipelineInfo).get();
}

ShaderParameter DelaunayTetRenderer::GetSceneParameter() {
	ShaderParameter sceneParams = {};
	sceneParams["vertices"] = (BufferParameter)vertices;
	sceneParams["colors"]   = (BufferParameter)colors;
	sceneParams["indices"]  = (BufferParameter)indices;
	sceneParams["spheres"]  = (BufferParameter)spheres;
	sceneParams["numTets"]  = (uint32_t)indices.size();
	sceneParams["aabbMin"] = minVertex;
	sceneParams["aabbMax"] = maxVertex;
	return sceneParams;
}

void DelaunayTetRenderer::ComputeSpheres(CommandContext& context, const ShaderParameter& scene) {
	ShaderParameter parameters = {};
	parameters["scene"] = scene;
	parameters["outputSpheres"] = (BufferParameter)spheres;

	context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)indices.size(), parameters);
	context.AddBarrier(spheres.SetState(Buffer::ResourceState{
		.stage = vk::PipelineStageFlagBits2::eComputeShader,
		.access = vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite,
		.queueFamily = context.QueueFamily()
	}));
}

void DelaunayTetRenderer::Sort(CommandContext& context, const ShaderParameter& scene, const float3 cameraPosition) {
	if (!sortPairs || sortPairs.size() != indices.size()) {
		sortPairs = Buffer::Create(context.GetDevice(), indices.size()*sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer);
	}
	if (reorderTets && (!sortedColors || sortedColors.size() != indices.size())) {
		sortedColors  = Buffer::Create(context.GetDevice(), indices.size()*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);
		sortedIndices = Buffer::Create(context.GetDevice(), indices.size()*sizeof(uint4),  vk::BufferUsageFlagBits::eStorageBuffer);
		sortedSpheres = Buffer::Create(context.GetDevice(), indices.size()*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);
	}

	ShaderParameter params = {};
	params["scene"] = scene;
	params["sortPairs"]     = (BufferParameter)sortPairs;
	params["sortedColors"]  = (BufferParameter)sortedColors;
	params["sortedIndices"] = (BufferParameter)sortedIndices;
	params["sortedSpheres"] = (BufferParameter)sortedSpheres;
	params["cameraPosition"] = cameraPosition;

	Pipeline& createSortPairs = *createSortPairsPipeline.get(context.GetDevice());
	auto descriptorSets = context.GetDescriptorSets(*createSortPairs.Layout());
	context.UpdateDescriptorSets(*descriptorSets, params, *createSortPairs.Layout());
	context.Dispatch(createSortPairs, (uint32_t)indices.size(), *descriptorSets);

	radixSort(context, sortPairs);

	if (reorderTets) context.Dispatch(*reorderTetPipeline.get(context.GetDevice()), (uint32_t)indices.size(), *descriptorSets);
}

void DelaunayTetRenderer::RenderProperties(CommandContext& context) {
	if (ImGui::CollapsingHeader("Camera")) {
		camera.DrawGui();
	}

	if (ImGui::CollapsingHeader("Scene")) {
		ImGui::Checkbox("Reorder on load", &reorderOnLoad);
		ImGui::Separator();
		ImGui::Text("%llu tetrahedra", indices.size());
		ImGui::Text("%llu vertices", vertices.size());
		ImGui::Separator();
		ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
		ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
		ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
		ImGui::Separator();
		Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);

	}

	if (ImGui::CollapsingHeader("Renderer")) {
		if (renderTarget) {
			ImGui::Text("%u x %u", renderTarget.Extent().x, renderTarget.Extent().y);
		}
		if (ImGui::BeginCombo("Mode", kRasterConfigurations[renderMode].name)) {
			for (uint32_t i = 0; i < std::ranges::size(kRasterConfigurations); i++) {
				if (ImGui::Selectable(kRasterConfigurations[i].name, renderMode == i)) {
					renderMode = i;
				}
			}
			ImGui::EndCombo();
		}

		ImGui::SliderFloat("Density threshold", &densityThreshold, 0.f, 1.f);
		ImGui::SliderFloat("% to draw", &percentTets, 0, 1);
		ImGui::Checkbox("Sorting", &enableSorting);
		if (enableSorting) {
			ImGui::Checkbox("Reordering", &reorderTets);
			//ImGui::SetTooltip("Reorder tet data after sorting. Avoids a layer of\nindirection at the cost of an extra copy step.");
		}
		if (kRasterConfigurations[renderMode].polygonMode == vk::PolygonMode::ePoint) {
			Gui::ScalarField("Point size", &pointSize, 0.1f, 0.f, 0.1f);
		} else {
			ImGui::Checkbox("Wireframe", &wireframe);
		}
	}
}

void DelaunayTetRenderer::RenderWidget(CommandContext& context, const double dt) {
	const float2 extentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
	const uint2 extent = uint2(extentf);
	if (extent.x == 0 || extent.y == 0) return;

	if (!renderTarget || renderTarget.Extent().x != extent.x || renderTarget.Extent().y != extent.y) {
		renderTarget = ImageView::Create(
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

			depthBuffer = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR16Sfloat,
					.extent = uint3(extent, 1),
					.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
					.queueFamilies = { context.QueueFamily() } }),
				vk::ImageSubresourceRange{
					.aspectMask = vk::ImageAspectFlagBits::eColor,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 });
	}

	// Draw the renderTarget image to the window
	ImGui::Image(Gui::GetTextureID(renderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(extentf));

	camera.Update(dt);

	// render the scene into renderTarget
	Render(context);
}

void DelaunayTetRenderer::Render(CommandContext& context) {
	const uint3 extent = renderTarget.Extent();

	if (!indices) {
		context.ClearColor(renderTarget, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 0 }});
		return;
	}

	context.PushDebugLabel("DelaunayTetRenderer::Render");

	ShaderParameter sceneParams = GetSceneParameter();

	float4x4 sceneToWorld = glm::translate(sceneTranslation) * glm::toMat4(glm::quat(sceneRotation)) * float4x4(sceneScale);
	float4x4 worldToScene = inverse(sceneToWorld);

	float3 rayOrigin = (float3)(worldToScene * float4(camera.position, 1));

	// sort
	if (enableSorting) {
		context.PushDebugLabel("Sort");
		Sort(context, sceneParams, (float3)(worldToScene * float4(camera.position, 1)));
		if (reorderTets) {
			sceneParams["colors"]  = (BufferParameter)sortedColors;
			sceneParams["indices"] = (BufferParameter)sortedIndices;
			sceneParams["spheres"] = (BufferParameter)sortedSpheres;
		}
		context.PopDebugLabel();
	}

	if (kRasterConfigurations[renderMode].fs == std::string("fs_depth"))
		context.ClearColor(depthBuffer, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 0 }});

	// rasterize tets
	{
		context.PushDebugLabel("DelaunayTetRenderer::Draw");

		Pipeline& renderPipeline = *GetRenderPipeline(context);

		auto descriptorSets = context.GetDescriptorSets(*renderPipeline.Layout());

		uint32_t tetCount = (uint32_t)(percentTets*indices.size());

		// prepare draw parameters
		{
			float aspect = (float)extent.x / (float)extent.y;
			float4x4 cameraToWorld = camera.GetCameraToWorld();
			float4x4 projection = camera.GetProjection(aspect);

			float4x4 sceneToCamera = inverse(cameraToWorld) * sceneToWorld;

			ShaderParameter params = {};
			params["scene"] = sceneParams;
			params["sortBuffer"] = (BufferParameter)sortPairs;
			params["depthBuffer"]= ImageParameter{ .image = depthBuffer, .imageLayout = vk::ImageLayout::eGeneral };
			params["view"] = sceneToCamera;
			params["projection"] = projection;
			params["viewProjection"] = projection * sceneToCamera;
			params["invProjection"] = inverse(projection);
			params["outputResolution"] = float2(extent.x, extent.y);
			params["cameraRotation"] = glm::toQuat(worldToScene * glm::toMat4(camera.Rotation()));
			params["rayOrigin"] = rayOrigin;
			params["densityScale"] = densityScale;
			params["densityThreshold"] = densityThreshold;
			params["pointSize"] = pointSize;

			context.UpdateDescriptorSets(*descriptorSets, params, *renderPipeline.Layout());
		}

		// rasterize scene

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
			.renderArea = vk::Rect2D{ {0, 0}, {extent.x, extent.y} },
			.layerCount = 1,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = attachments });

		context->setViewport(0, vk::Viewport{ 0, 0, (float)extent.x, (float)extent.y, 0, 1});
		context->setScissor(0, vk::Rect2D{ {0, 0}, {extent.x, extent.y}});

		if (tetCount > 0) {
			context->bindPipeline(vk::PipelineBindPoint::eGraphics, **renderPipeline);
			context.BindDescriptors(*renderPipeline.Layout(), *descriptorSets);

			if (kRasterConfigurations[renderMode].polygonMode == vk::PolygonMode::ePoint)
				context->draw(tetCount, 1, 0, 0); // 1 vert per tet
			else {
				context->draw(12, tetCount, 0, 0); // 4 tris per tet x 3 verts per tri -> 12 vertices per tet
			}
		}

		context->endRendering();

		context.PopDebugLabel();
	}

	// compute alpha = 1 - T
	{
		ShaderParameter params = {};
		params["image"] = ImageParameter{ .image = renderTarget, .imageLayout = vk::ImageLayout::eGeneral };
		params["dim"] = (uint2)extent;
		context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), (uint2)extent, params);
	}

	context.PopDebugLabel();
}