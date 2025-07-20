#include <portable-file-dialogs.h>
#include <tinyply.h>

#include <Rose/Core/DxgiFormatConvert.h>
#include <Rose/Core/Gui.hpp>
#include "TetrahedronScene.hpp"

// matches the value in EvaluateSH.cs.slang
#define COEFFS_PER_BUF 8

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
	int32_t minSH = std::numeric_limits<int32_t>::max();
	int32_t maxSH = -1;
	for (const auto& prop : tet_element->properties) {
		if (!prop.name.starts_with("sh_"))
			continue;
		int32_t sh_i;
		if (sscanf(prop.name.c_str(), "sh_%d", &sh_i) > 0)
		{
			minSH = min(minSH, sh_i);
			maxSH = max(maxSH, sh_i);
		}
	}
	if (maxSH == -1) {
		std::cerr << "No colors in ply file." << std::endl;
		return;
	}

	// parse SH coefficient properties
	std::vector<std::vector<std::string>> sh_props;
	for (uint32_t i = minSH; i <= maxSH; i++) {
		const auto prefix = "sh_" + std::to_string(i);
		const uint32_t bufId = (i - minSH) / COEFFS_PER_BUF;
		if (bufId <= sh_props.size()) sh_props.resize(bufId + 1);
		sh_props[bufId].emplace_back(prefix + "_r");
		sh_props[bufId].emplace_back(prefix + "_g");
		sh_props[bufId].emplace_back(prefix + "_b");
	}
	
	auto getPlyData = []<typename T>(const auto& ply_data) -> std::span<T>{
		return std::span{reinterpret_cast<T*>(ply_data->buffer.get()), ply_data->buffer.size_bytes()/sizeof(T)};
	};

	auto ply_vertices      = ply.request_properties_from_element("vertex", { "x", "y", "z" });
	auto ply_tet_indices   = ply.request_properties_from_element("tetrahedron", { "vertex_indices" }, 4);
	auto ply_tet_densities = ply.request_properties_from_element("tetrahedron", { "s" });
	auto ply_tet_gradients = ply.request_properties_from_element("tetrahedron", { "grd_x", "grd_y", "grd_z" });
	std::vector<std::shared_ptr<tinyply::PlyData>> ply_vertex_sh(sh_props.size());
	for (uint32_t i = 0; i < sh_props.size(); i++)
		ply_vertex_sh[i] = ply.request_properties_from_element("tetrahedron", sh_props[i]);

	ply.read(file);

	std::span pos  = getPlyData.template operator()<float3>(ply_vertices);
	std::span inds = getPlyData.template operator()<uint4>(ply_tet_indices);
	std::span dens = getPlyData.template operator()<float>(ply_tet_densities);
	std::span grad = getPlyData.template operator()<float3>(ply_tet_gradients);

	const uint32_t numTets = (uint32_t)inds.size();

	numTetSHCoeffs = 0;
	std::vector<std::span<float3>> sh(ply_vertex_sh.size());
	for (uint32_t i = 0; i < ply_vertex_sh.size(); i++) {
		sh[i] = getPlyData.template operator()<float3>(ply_vertex_sh[i]);
		numTetSHCoeffs += sh[i].size() / numTets;
	}

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
	bool compressSH = false;
	
	vertices         = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer);
	tetIndices       = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	tetGradients     = context.UploadData(grad, vk::BufferUsageFlagBits::eStorageBuffer);
	tetOffsets     = Buffer::Create(context.GetDevice(), numTets*sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
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

		tetSH.resize(sh.size());
		for (uint32_t i = 0; i < sh.size(); i++)
		{
			if (compressSH)
			{
				// compress SH coefficients to float16
				tetSH[i] = Buffer::Create(context.GetDevice(), sh[i].size()*sizeof(uint16_t)*3, vk::BufferUsageFlagBits::eStorageBuffer);

				const uint32_t n = (uint32_t)(sh.size()*3);
				ShaderParameter parameters;
				parameters["inputData"]  = (BufferParameter)context.UploadData(sh[i], vk::BufferUsageFlagBits::eStorageBuffer);
				parameters["outputData"] = (BufferParameter)tetSH[i];
				parameters["count"] = n;
				context.Dispatch(f32tof16pipeline, n, parameters);
			}
			else
			{
				tetSH[i] = context.UploadData(sh[i], vk::BufferUsageFlagBits::eStorageBuffer);
			}
		}
	}

	{
		ShaderParameter parameters = {};
		parameters["scene"] = GetShaderParameter();
		parameters["outputSpheres"] = (BufferParameter)tetCircumspheres;
		parameters["outputCentroids"] = (BufferParameter)tetCentroids;
		parameters["outputOffsets"] = (BufferParameter)tetOffsets;
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
		totalSize += tetIndices      .size_bytes();
		totalSize += tetDensities    .size_bytes();
		totalSize += tetGradients    .size_bytes();
		totalSize += tetCircumspheres.size_bytes();
		for (const auto& sh : tetSH) totalSize += sh.size_bytes();
		const auto[x, unit] = FormatBytes(totalSize);
		ImGui::Text("%lu%s", x, unit);
	}

	if (vertices)
		ImGui::Text("SH coeffs: %u", numTetSHCoeffs);

	ImGui::Separator();
	ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
	ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
	ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
	ImGui::Separator();
	Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);
}
