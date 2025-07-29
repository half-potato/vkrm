#pragma once

#include "../RenderContext.hpp"

namespace vkDelTet {

class MeshShaderRenderer { 
private:
    bool  wireframe = false;
    float percentTets = 1.f; // percent of tets to draw
    float densityThreshold = 0.f;

    TexelBufferView vertexColors;

    PipelineCache renderPipeline = PipelineCache({
        { FindShaderPath("MeshShaderRenderer.3d.slang"), "meshmain" },
        { FindShaderPath("MeshShaderRenderer.3d.slang"), "fsmain" }
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
    inline const char* Name() const { return "Mesh shader"; }
    inline const char* Description() const { return "Rasterize with mesh shader"; }

	void DrawGui(CommandContext& context) {
		ImGui::SliderFloat("Density threshold", &densityThreshold, 0.f, 1.f);
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
        
        ShaderParameter sceneParams = renderContext.scene.GetShaderParameter();
        
        renderContext.PrepareRender(context, rayOrigin, false);

        context.PushDebugLabel("Rasterize");

        Pipeline& pipeline = GetPipeline(context, renderContext);
        auto descriptorSets = context.GetDescriptorSets(*pipeline.Layout());

        // prepare draw parameters
        {
            ShaderParameter params = {};
            params["scene"] = sceneParams;
            params["sortPayloads"] = (BufferParameter)renderContext.sortPayloads;
            // params["sortBuffer"] = (BufferParameter)renderContext.sortBuffer;
            params["tetColors"]        = (BufferParameter)renderContext.evaluatedColors;
            params["viewProjection"] = viewProjection;
            params["rayOrigin"] = rayOrigin;
            params["densityThreshold"] = densityThreshold * renderContext.scene.DensityScale();
            for (uint32_t i = 0; i < renderContext.scene.TetSH().size(); i++)
                    params["shCoeffs"][i] = (BufferParameter)renderContext.scene.TetSH()[i];
            params["tetCentroids"]    = (BufferParameter)renderContext.scene.TetCentroids();
            params["tetOffsets"]    = (BufferParameter)renderContext.scene.TetOffsets();

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

            const uint32_t tetsPerGroup = pipeline.GetShader()->WorkgroupSize().x/4;
            // context->drawMeshTasksEXT((tetCount + tetsPerGroup-1) / tetsPerGroup, 1, 1);
            context->drawMeshTasksIndirectEXT(
                **renderContext.meshDrawArgs.mBuffer,  // The buffer with the mesh task args
                renderContext.meshDrawArgs.mOffset,    // Offset into the buffer
                1,                                     // drawCount (usually 1)
                sizeof(vk::DrawMeshTasksIndirectCommandEXT) // stride
            );
        }

        renderContext.EndRendering(context);

        context.PopDebugLabel();
    }
};

}
