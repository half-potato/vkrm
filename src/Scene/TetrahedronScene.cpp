#include <portable-file-dialogs.h>
#include <tinyply.h>

#include <Rose/Core/DxgiFormatConvert.h>
#include "TetrahedronScene.hpp"

using namespace vkDelTet;

void TetrahedronScene::ComputeSpheres(CommandContext& context) {
	ShaderParameter parameters = {};
	parameters["scene"] = GetShaderParameter();
	parameters["outputSpheres"] = (BufferParameter)spheres;

	context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)spheres.size(), parameters);
}

void TetrahedronScene::Load(CommandContext& context, const std::filesystem::path& p) {
	if (!std::filesystem::exists(p))
		return;

	std::ifstream file;
	file.open(p, std::ios::binary);

	tinyply::PlyFile ply;
	ply.parse_header(file);
	const auto elements = ply.get_elements();

	if (std::ranges::find(elements, "vertex", &tinyply::PlyElement::name) == elements.end()) {
		std::cerr << "No vertex element in ply file." << std::endl;
		return;
	}

	auto tet_it = std::ranges::find(elements, "tetrahedron", &tinyply::PlyElement::name);
	if (tet_it == elements.end()) {
		std::cerr << "No tetrahedron element in ply file." << std::endl;
		return;
	}

	// find light properties, if any
	std::vector<std::shared_ptr<tinyply::PlyData>> ply_lights;
	const auto& tet_element = *tet_it;
	for (const auto& prop : tet_element.properties) {
		if (prop.name[0] != 'l')
			continue;
		const size_t o = prop.name.find('_');
		if (o == std::string::npos)
			continue;
		
		const uint32_t lightIndex = std::stoi(prop.name.substr(1, o-1));
		if (ply_lights.size() <= lightIndex)
			ply_lights.resize(lightIndex+1);
	}

	numLights = (uint32_t)ply_lights.size();

	std::vector<std::string> lightColorProps;
	std::vector<std::string> lightDirectionProps;
	lightColorProps.reserve(numLights*4);
	lightDirectionProps.reserve(numLights*2);
	for (uint32_t i = 0; i < ply_lights.size(); i++) {
		const auto s = std::to_string(i);
		lightColorProps.emplace_back("l" + s + "_r");
		lightColorProps.emplace_back("l" + s + "_g");
		lightColorProps.emplace_back("l" + s + "_b");
		lightColorProps.emplace_back("l" + s + "_roughness");
		lightDirectionProps.emplace_back("l" + s + "_phi");
		lightDirectionProps.emplace_back("l" + s + "_theta");
	}
	
	auto ply_vertices    = ply.request_properties_from_element("vertex",      { "x", "y", "z" });
	auto ply_tet_colors  = ply.request_properties_from_element("tetrahedron", { "r", "g", "b", "s" });
	auto ply_tet_indices = ply.request_properties_from_element("tetrahedron", { "vertex_indices" }, 4);
	auto ply_tet_light_colors     = numLights == 0 ? std::shared_ptr<tinyply::PlyData>{} : ply.request_properties_from_element("tetrahedron", lightColorProps);
	auto ply_tet_light_directions = numLights == 0 ? std::shared_ptr<tinyply::PlyData>{} : ply.request_properties_from_element("tetrahedron", lightDirectionProps);

	ply.read(file);

	std::span pos  {reinterpret_cast<float3*>(ply_vertices->buffer.get()),    ply_vertices->buffer.size_bytes()/sizeof(float3)};
	std::span col  {reinterpret_cast<float4*>(ply_tet_colors ->buffer.get()), ply_tet_colors ->buffer.size_bytes()/sizeof(float4)};
	std::span inds {reinterpret_cast<uint4* >(ply_tet_indices->buffer.get()), ply_tet_indices->buffer.size_bytes()/sizeof(uint4)};
	std::span lightCol = numLights == 0 ? std::span<float4>{} : std::span<float4>{reinterpret_cast<float4*>(ply_tet_light_colors->buffer.get()),     ply_tet_light_colors->buffer.size_bytes()/sizeof(float4)};
	std::span lightDir = numLights == 0 ? std::span<float2>{} : std::span<float2>{reinterpret_cast<float2*>(ply_tet_light_directions->buffer.get()), ply_tet_light_directions->buffer.size_bytes()/sizeof(float2)};

	const uint32_t numTets = (uint32_t)inds.size();

	if (col.size() != numTets) {
		std::cerr << "Size of color buffer " << col.size() << " != " << numTets << "!" << std::endl;
		return;
	}
 	if ((lightCol.size() > 0 && lightCol.size()/numLights != numTets) || lightCol.size() != lightDir.size()) {
		std::cerr << "Size of light buffer " << lightColorProps.size()/numLights << " != " << numTets << "!" << std::endl;
		return;
	}

	// pre-quantization: scale densities by tet radii
	maxDensity = 0;
	for (const float4 c : col)
		maxDensity = max(maxDensity, c.w);
	
	context.GetDevice().Wait(); // wait in case previous vertices/colors/indices are in use still
	
	vertices  = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer);
	indices   = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	colors    = TexelBufferView::Create(context.GetDevice(), Buffer::Create(context.GetDevice(), numTets*sizeof(uint32_t), vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eA2R10G10B10UnormPack32);
	densities = TexelBufferView::Create(context.GetDevice(), Buffer::Create(context.GetDevice(), numTets*sizeof(uint16_t), vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eR16Sfloat);
	spheres   = Buffer::Create(context.GetDevice(), numTets*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);
	if (numLights > 0) {
		lightColors     = TexelBufferView::Create(context.GetDevice(), context.UploadData(lightCol, vk::BufferUsageFlagBits::eUniformTexelBuffer), vk::Format::eR32G32B32A32Sfloat);
		lightDirections = TexelBufferView::Create(context.GetDevice(), context.UploadData(lightDir, vk::BufferUsageFlagBits::eUniformTexelBuffer), vk::Format::eR32G32Sfloat);
	}

	minVertex = float3( FLT_MAX );
	maxVertex = float3( FLT_MIN );
	for (float3 p : pos) {
		minVertex = min(p, minVertex);
		maxVertex = max(p, maxVertex);
	}

	// compress colors and densities
	{
		Pipeline& pipeline = *compressColorsPipeline.get(context.GetDevice());
		ShaderParameter parameters;
		parameters["input"]   = (BufferParameter)context.UploadData(col, vk::BufferUsageFlagBits::eStorageBuffer);;
		parameters["output"]  = (BufferParameter)colors.GetBuffer();
		parameters["outputW"] = (BufferParameter)densities.GetBuffer();
		parameters["size"] = numTets;
		context.Dispatch(pipeline, numTets, parameters);
	}

	ComputeSpheres(context);
}

ShaderParameter TetrahedronScene::GetShaderParameter() {
	ShaderParameter sceneParams = {};
	sceneParams["vertices"]  = (BufferParameter)vertices;
	sceneParams["colors"]    = (TexelBufferParameter)colors;
	sceneParams["densities"] = (TexelBufferParameter)densities;
	sceneParams["indices"]   = (BufferParameter)indices;
	sceneParams["lightColors"]     = (TexelBufferParameter)lightColors;
	sceneParams["lightDirections"] = (TexelBufferParameter)lightDirections;
	sceneParams["spheres"]   = (BufferParameter)spheres;
	sceneParams["numTets"]   = (uint32_t)indices.size();
	sceneParams["aabbMin"]      = minVertex;
	sceneParams["aabbMax"]      = maxVertex;
	sceneParams["densityScale"] = densityScale;
	sceneParams["numLights"]    = numLights;
	return sceneParams;
}

void TetrahedronScene::DrawGui(CommandContext& context) {
	{
		const auto[x, unit] = FormatNumber(spheres.size());
		ImGui::Text("%0.2f%s tetrahedra", x, unit);
	}
	{
		const auto[x, unit] = FormatNumber(vertices.size());
		ImGui::Text("%0.2f%s vertices", x, unit);
	}
	{
		const auto[x, unit] = FormatBytes(vertices.size_bytes() + colors.GetBuffer().size_bytes() + indices.size_bytes() + spheres.size_bytes());
		ImGui::Text("%lu%s", x, unit);
	}

	ImGui::Text("%lu lights per tet", numLights);

	ImGui::Separator();
	ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
	ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
	ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
	ImGui::Separator();
	Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);
}