#pragma once

#include "../RenderContext.hpp"

namespace vkDelTet {

class InstancedRenderer { 
private:
    bool  wireframe = false;
    float percentTets = 1.f; // percent of tets to draw
    float densityThreshold = 0.f;
    Mesh mesh = {};
    MeshLayout meshLayout = {};

    PipelineCache renderPipeline = PipelineCache({
        { FindShaderPath("InstancedRenderer.3d.slang"), "vsmain" },
        { FindShaderPath("InstancedRenderer.3d.slang"), "fsmain" }
    });

    inline Pipeline& GetPipeline(CommandContext& context, RenderContext& renderContext) {
        ShaderDefines defines {};
        mesh = Mesh {
                .indexBuffer = context.UploadData(std::vector<uint16_t>{ 
                    0, 2, 1, 
                    1, 2, 3, 
                    0, 3, 2, 
                    3, 0, 1
                }, vk::BufferUsageFlagBits::eIndexBuffer),
                .indexSize = sizeof(uint16_t),
                .topology = vk::PrimitiveTopology::eTriangleList };
        // mesh.vertexAttributes[MeshVertexAttributeType::ePosition].emplace_back(
        //         context.UploadData(std::vector<float3>{
        //                         float3(-.25f, -.25f, 0), float3(.25f, -.25f, 0),
        //                         float3(-.25f,  .25f, 0), float3(.25f,  .25f, 0),
        //                 }, vk::BufferUsageFlagBits::eVertexBuffer),
        //         MeshVertexAttributeLayout{
        //                 .stride = sizeof(float3),
        //                 .format = vk::Format::eR32G32B32Sfloat,
        //                 .offset = 0,
        //                 .inputRate = vk::VertexInputRate::eVertex });
        // auto &vertexShader = renderPipeline.GetShader(vk::ShaderStageFlagBits::eVertex);
        auto &vertexShader = renderPipeline.getShader(context.GetDevice(), 0, defines);
        meshLayout = mesh.GetLayout(*vertexShader);

        GraphicsPipelineInfo pipelineInfo {
            .vertexInputState = VertexInputDescription{
                .bindings = meshLayout.bindings,
                .attributes = meshLayout.attributes },
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
    inline const char* Name() const { return "InstancedRaster"; }
    inline const char* Description() const { return "Instanced Geometry Rasterization"; }

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

            context.UpdateDescriptorSets(*descriptorSets, params, *pipeline.Layout());
        }

        // rasterize scene

        renderContext.BeginRendering(context);       
        context->setViewport(0, vk::Viewport{ 0, 0, (float)extent.x, (float)extent.y, 0, 1});
        context->setScissor(0,  vk::Rect2D{ {0, 0}, { extent.x, extent.y }});

        context->bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);
        context.BindDescriptors(*pipeline.Layout(), *descriptorSets);
        // context->draw(tetCount*12, 1, 0, 0);
        mesh.Bind(context, meshLayout);
        context->drawIndexedIndirect(
            **renderContext.insDrawArgs.mBuffer,  // Dereference the handle from the wrapper
            renderContext.insDrawArgs.mOffset,    // Provide the buffer offset
            1,                                     // drawCount
            sizeof(vk::DrawIndirectCommand)        // stride
        );

        renderContext.EndRendering(context);

        context.PopDebugLabel();
    }
};

}

