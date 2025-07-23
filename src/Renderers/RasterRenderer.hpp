#pragma once

#include "../RenderContext.hpp"

namespace vkDelTet {

class RasterRenderer { 
private:
    bool  wireframe = false;
    float percentTets = 1.f; // percent of tets to draw
    float densityThreshold = 0.f;

    PipelineCache renderPipeline = PipelineCache({
        { FindShaderPath("RasterRenderer.3d.slang"), "vsmain" },
        { FindShaderPath("RasterRenderer.3d.slang"), "fsmain" }
    });

    inline Pipeline& GetPipeline(CommandContext& context, RenderContext& renderContext) {
        ShaderDefines defines {};

        GraphicsPipelineInfo pipelineInfo {
            .vertexInputState = VertexInputDescription{},
            .inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
                .topology = vk::PrimitiveTopology::eTriangleList },
            .rasterizationState = vk::PipelineRasterizationStateCreateInfo{
                .depthClampEnable = false,
                .rasterizerDiscardEnable = false,
                .polygonMode = wireframe ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
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
    inline const char* Name() const { return "HW Raster"; }
    inline const char* Description() const { return "HW Rasterization"; }

    void DrawGui(CommandContext& context) {
        ImGui::SliderFloat("Density threshold", &densityThreshold, 0.f, 1.0);
        ImGui::SliderFloat("% to draw", &percentTets, 0, 1);

        ImGui::Checkbox("Wireframe", &wireframe);
    }

    void Render(CommandContext& context, RenderContext& renderContext) {
        const uint2    extent = (uint2)renderContext.renderTarget.Extent();
        const float4x4 cameraToWorld = renderContext.camera.GetCameraToWorld();
        const float4x4 sceneToWorld  = renderContext.scene.Transform();
        const float4x4 worldToScene  = inverse(sceneToWorld);
        const float4x4 sceneToCamera = inverse(cameraToWorld) * sceneToWorld;
        const float4x4 projection = renderContext.camera.GetProjection((float)extent.x / (float)extent.y);
        const float4x4 viewProjection = projection * sceneToCamera;
        const float3   rayOrigin = (float3)(worldToScene * float4(renderContext.camera.position, 1));

        renderContext.PrepareRender(context, rayOrigin);

        context.PushDebugLabel("Rasterize");

        Pipeline& pipeline = GetPipeline(context, renderContext);
        auto descriptorSets = context.GetDescriptorSets(*pipeline.Layout());

        // prepare draw parameters
        {
            ShaderParameter params = {};
            params["scene"]            = renderContext.scene.GetShaderParameter();
            params["sortPayloads"]     = (BufferParameter)renderContext.sortPayloads;
            params["tetColors"]        = (BufferParameter)renderContext.evaluatedColors;
            params["viewProjection"]   = viewProjection;
            params["invProjection"]    = inverse(viewProjection);
            params["rayOrigin"]        = rayOrigin;
            params["densityThreshold"] = densityThreshold * renderContext.scene.DensityScale();
            params["outputResolution"] = (float2)extent;
            params["visibleTets"] = (BufferParameter)renderContext.visibleTets;
            params["blockSumAtomicCounter"] = (BufferParameter)renderContext.blockSumAtomicCounter;

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
            // context->draw(tetCount*12, 1, 0, 0);
            context->drawIndirect(
                **renderContext.drawArgs.mBuffer,  // Dereference the handle from the wrapper
                renderContext.drawArgs.mOffset,    // Provide the buffer offset
                1,                                     // drawCount
                sizeof(vk::DrawIndirectCommand)        // stride
            );

        }

        renderContext.EndRendering(context);

        context.PopDebugLabel();
    }
};

}
