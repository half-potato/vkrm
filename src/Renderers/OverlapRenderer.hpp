#pragma once

#include "../RenderContext.hpp"

namespace vkDelTet {

class BillboardRenderer { 
private:
    float percentTets = 1.f; // percent of tets to draw
    float densityThreshold = 0.f;

    PipelineCache renderPipeline = PipelineCache({
        { FindShaderPath("OverlapRenderer.slang"), "vsmain" },
        { FindShaderPath("OverlapRenderer.slang"), "fsmain" }
    });

    inline Pipeline& GetPipeline(CommandContext& context, RenderContext& renderContext) {
        ShaderDefines defines;

        GraphicsPipelineInfo pipelineInfo {
            .vertexInputState = VertexInputDescription{},
            .inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
                .topology = vk::PrimitiveTopology::ePointList },
            .rasterizationState = vk::PipelineRasterizationStateCreateInfo{
                .depthClampEnable = false,
                .rasterizerDiscardEnable = false,
                .polygonMode = vk::PolygonMode::ePoint,
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
                .attachments = { renderContext.GetBlendState() } },
            .dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
            .dynamicRenderingState = DynamicRenderingState{
                .colorFormats = { renderContext.renderTarget.GetImage()->Info().format } } };

        return *renderPipeline.get(context.GetDevice(), defines, pipelineInfo).get();
    }

public:
    inline const char* Name() const { return "Billboards"; }
    inline const char* Description() const { return "Draw billboards (imposters) at tet centroids, and compute intersection in fragment shader"; }

	void DrawGui(CommandContext& context) {
		ImGui::SliderFloat("Density threshold", &densityThreshold, 0.f, 1.f);
		ImGui::SliderFloat("% to draw", &percentTets, 0, 1);
    }

	void Render(CommandContext& context, RenderContext& renderContext) {
        const float4x4 cameraToWorld = renderContext.camera.GetCameraToWorld();
        const float4x4 sceneToWorld  = renderContext.scene.Transform();
        const float4x4 worldToScene  = inverse(sceneToWorld);
        const float4x4 sceneToCamera = inverse(cameraToWorld) * sceneToWorld;
        const float3   rayOrigin = (float3)(worldToScene * float4(renderContext.camera.position, 1));

        const ShaderParameter sceneParams = renderContext.scene.GetShaderParameter();

        renderContext.PrepareRender(context, rayOrigin);

        context.PushDebugLabel("Rasterize");

        Pipeline& pipeline = GetPipeline(context, renderContext);
        auto descriptorSets = context.GetDescriptorSets(*pipeline.Layout());
        const uint2 extent = (uint2)renderContext.renderTarget.Extent();

        // prepare draw parameters
        {
            const float4x4 projection = renderContext.camera.GetProjection((float)extent.x / (float)extent.y);

            ShaderParameter params = {};
            params["scene"] = sceneParams;
            params["tetColors"]        = (BufferParameter)renderContext.evaluatedColors;
            params["sortPayloads"]   = (BufferParameter)renderContext.sortPayloads;
            params["viewProjection"] = projection * sceneToCamera;
            params["invProjection"] = inverse(projection * sceneToCamera);
            params["cameraRotation"] = glm::toQuat(worldToScene * glm::toMat4(renderContext.camera.GetRotation()));
            params["rayOrigin"] = rayOrigin;
            params["densityThreshold"] = densityThreshold * renderContext.scene.DensityScale() * renderContext.scene.MaxDensity();
            params["outputResolution"] = (float2)extent;

            context.UpdateDescriptorSets(*descriptorSets, params, *pipeline.Layout());
        }

        // rasterize scene

        renderContext.BeginRendering(context);       
        context->setViewport(0, vk::Viewport{ 0, 0, (float)extent.x, (float)extent.y, 0, 1});
        context->setScissor(0,  vk::Rect2D{ {0, 0}, { extent.x, extent.y }});

        uint32_t tetCount = (uint32_t)(percentTets*renderContext.scene.TetCount());
        if (tetCount > 0) {
            context->bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);
            context.BindDescriptors(*pipeline.Layout(), *descriptorSets);
            context->draw(tetCount, 1, 0, 0); // 1 vert per tet
        }

        renderContext.EndRendering(context);

        context.PopDebugLabel();
    }
};

}

