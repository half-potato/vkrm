#include <portable-file-dialogs.h>
#include <tinyply.h>

#include <Rose/Core/DxgiFormatConvert.h>
#include <Rose/Core/Gui.hpp>
#include "TetrahedronScene.hpp"

using namespace vkDelTet;

void TetrahedronScene::Load(CommandContext& context, const std::filesystem::path& p) {
	if (!std::filesystem::exists(p))
		return;

	std::ifstream file;
	file.open(p, std::ios::binary);

	tinyply::PlyFile ply;
	ply.parse_header(file);
	const auto elements = ply.get_elements();

	auto tet_element = std::ranges::find(elements, "tetrahedron", &tinyply::PlyElement::name);
	if (tet_element == elements.end()) {
		std::cerr << "No vertex element in ply file." << std::endl;
		return;
	}
	if (std::ranges::find(elements, "tetrahedron", &tinyply::PlyElement::name) == elements.end()) {
		std::cerr << "No tetrahedron element in ply file." << std::endl;
		return;
	}

	// find number of sh coeffs
	uint32_t sh_n = 0;
	for (const auto& prop : tet_element->properties) {
		if (!prop.name.starts_with("sh_"))
			continue;
		uint32_t sh_i;
		if (sscanf(prop.name.c_str(), "sh_%u", &sh_i) > 0)
			sh_n = max(sh_n, sh_i+1);
	}

	if (sh_n == 0) {
		std::cerr << "No colors in ply file." << std::endl;
		return;
	}

	// parse SH coefficient properties
	std::vector<std::string> sh_props;
	for (uint32_t i = 1; i < sh_n; i++) {
		const auto li = "sh_" + std::to_string(i) + "_";
		sh_props.emplace_back(li + "r");
		sh_props.emplace_back(li + "g");
		sh_props.emplace_back(li + "b");
	}
	
	auto getPlyData = []<typename T>(const auto& ply_data) -> std::span<T>{
		return std::span{reinterpret_cast<T*>(ply_data->buffer.get()), ply_data->buffer.size_bytes()/sizeof(T)};
	};

	auto ply_vertices      = ply.request_properties_from_element("vertex", { "x", "y", "z" });
	auto ply_vertex_sh     = ply.request_properties_from_element("tetrahedron", sh_props);
	auto ply_tet_indices   = ply.request_properties_from_element("tetrahedron", { "vertex_indices" }, 4);
	auto ply_tet_densities = ply.request_properties_from_element("tetrahedron", { "s" });
	auto ply_tet_gradients = ply.request_properties_from_element("tetrahedron", { "grd_x", "grd_y", "grd_z" });

	ply.read(file);

	std::span pos  = getPlyData.template operator()<float3>(ply_vertices);
	std::span sh   = getPlyData.template operator()<float3>(ply_vertex_sh);
	std::span inds = getPlyData.template operator()<uint4>(ply_tet_indices);
	std::span dens = getPlyData.template operator()<float>(ply_tet_densities);
	std::span grad = getPlyData.template operator()<float3>(ply_tet_gradients);

	const uint32_t numTets = (uint32_t)inds.size();

	numTetSHCoeffs = sh.size() / numTets;

	minVertex = float3( FLT_MAX );
	maxVertex = float3( FLT_MIN );
	for (float3 p : pos) {
		minVertex = min(p, minVertex);
		maxVertex = max(p, maxVertex);
	}
	maxDensity = 0;
	for (const float d : dens)
		maxDensity = max(maxDensity, d);

	context.GetDevice().Wait(); // wait in case previous vertices/colors/indices are in use still

	bool compressDensities = false;
	bool compressVertexSH = false;
	
	vertices         = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer);
	tetIndices       = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	tetGradients     = context.UploadData(grad, vk::BufferUsageFlagBits::eStorageBuffer);
	tetCentroids     = Buffer::Create(context.GetDevice(), numTets*sizeof(float3), vk::BufferUsageFlagBits::eStorageBuffer);
	tetCircumspheres = Buffer::Create(context.GetDevice(), numTets*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);
	
	{
		Pipeline& f32tof16pipeline = *compressColorsPipeline.get(context.GetDevice(), {
			{ "INPUT_TYPE",  "float" },
			{ "OUTPUT_TYPE", "uint16_t" },
			{ "COMPRESS_FN", "(uint16_t)f32tof16(i)" },
		});
		
		if (compressDensities)
		{
			tetDensities = TexelBufferView::Create(context.GetDevice(), Buffer::Create(context.GetDevice(), numTets*sizeof(uint16_t), vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eR16Sfloat);

			// compress densities to float16
			ShaderParameter parameters;
			parameters["inputData"]  = (BufferParameter)context.UploadData(dens, vk::BufferUsageFlagBits::eStorageBuffer);
			parameters["outputData"] = (BufferParameter)tetDensities.GetBuffer();
			parameters["count"] = numTets;
			context.Dispatch(f32tof16pipeline, numTets, parameters);
		}
		else
		{
			tetDensities = TexelBufferView::Create(context.GetDevice(), context.UploadData(dens, vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eR32Sfloat);
		}

		if (compressVertexSH)
		{
			// compress SH coefficients to float16
			tetSH = Buffer::Create(context.GetDevice(), sh.size()*sizeof(uint16_t)*3, vk::BufferUsageFlagBits::eStorageBuffer);

			const uint32_t n = (uint32_t)(sh.size()*3);
			ShaderParameter parameters;
			parameters["inputData"]  = (BufferParameter)context.UploadData(sh, vk::BufferUsageFlagBits::eStorageBuffer);
			parameters["outputData"] = (BufferParameter)tetSH;
			parameters["count"] = n;
			context.Dispatch(f32tof16pipeline, n, parameters);
		}
		else
		{
			tetSH = context.UploadData(sh, vk::BufferUsageFlagBits::eStorageBuffer);
		}
	}

	{
		ShaderParameter parameters = {};
		parameters["scene"] = GetShaderParameter();
		parameters["outputSpheres"] = (BufferParameter)tetCircumspheres;
		parameters["outputCentroids"] = (BufferParameter)tetCentroids;
		context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)tetCircumspheres.size(), parameters);
	}
}

ShaderParameter TetrahedronScene::GetShaderParameter() {
	ShaderParameter sceneParams = {};
	sceneParams["vertices"]     = (BufferParameter)vertices;
	sceneParams["tetDensities"] = (TexelBufferParameter)tetDensities;
	sceneParams["tetIndices"]   = (BufferParameter)tetIndices;
	sceneParams["tetGradients"] = (BufferParameter)tetGradients;
	sceneParams["aabbMin"]      = minVertex;
	sceneParams["aabbMax"]      = maxVertex;
	sceneParams["densityScale"] = densityScale;
	sceneParams["numTets"]      = TetCount();
	sceneParams["numVertices"]  = (uint32_t)vertices.size();
	return sceneParams;
}

void TetrahedronScene::DrawGui(CommandContext& context) {
	{
		const auto[x, unit] = FormatNumber(TetCount());
		ImGui::Text("%0.2f%s tetrahedra", x, unit);
	}
	{
		const auto[x, unit] = FormatNumber(vertices.size());
		ImGui::Text("%0.2f%s vertices", x, unit);
	}
	{	
		size_t totalSize = 0;
		totalSize += vertices        .size_bytes();
		totalSize += tetSH           .size_bytes();
		totalSize += tetIndices      .size_bytes();
		totalSize += tetDensities    .size_bytes();
		totalSize += tetGradients    .size_bytes();
		totalSize += tetCircumspheres.size_bytes();
		const auto[x, unit] = FormatBytes(totalSize);
		ImGui::Text("%lu%s", x, unit);
	}

	if (vertices && tetSH)
		ImGui::Text("SH coeffs: %u", numTetSHCoeffs);

	ImGui::Separator();
	ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
	ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
	ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
	ImGui::Separator();
	Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);
}