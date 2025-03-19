#include <portable-file-dialogs.h>
#include <tinyply.h>

#include <Rose/Core/DxgiFormatConvert.h>
#include <Rose/Core/Gui.hpp>
#include "TetrahedronScene.hpp"

using namespace vkDelTet;

void TetrahedronScene::ComputeSpheres(CommandContext& context) {
	ShaderParameter parameters = {};
	parameters["scene"] = GetShaderParameter();
	parameters["outputSpheres"] = (BufferParameter)tetCircumspheres;

	context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)tetCircumspheres.size(), parameters);
}

void TetrahedronScene::Load(CommandContext& context, const std::filesystem::path& p) {
	if (!std::filesystem::exists(p))
		return;

	std::ifstream file;
	file.open(p, std::ios::binary);

	tinyply::PlyFile ply;
	ply.parse_header(file);
	const auto elements = ply.get_elements();

	auto vert_it = std::ranges::find(elements, "vertex", &tinyply::PlyElement::name);
	if (vert_it == elements.end()) {
		std::cerr << "No vertex element in ply file." << std::endl;
		return;
	}
	if (std::ranges::find(elements, "tetrahedron", &tinyply::PlyElement::name) == elements.end()) {
		std::cerr << "No tetrahedron element in ply file." << std::endl;
		return;
	}

	vertexSHDegree = 0;
	for (const auto& prop : vert_it->properties) {
		if (!prop.name.starts_with("sh_"))
			continue;
		uint32_t sh_i;
		if (sscanf(prop.name.c_str(), "sh_%u", &sh_i) > 0)
			vertexSHDegree = max(vertexSHDegree, sh_i+1);
	}

	if (vertexSHDegree == 0) {
		std::cerr << "No vertex colors in ply file." << std::endl;
		return;
	}

	std::vector<std::string> vertexSHProps;
	vertexSHProps.reserve(vertexSHDegree*3);
	for (uint32_t i = 0; i < vertexSHDegree; i++) {
		const auto li = "sh_" + std::to_string(i) + "_";
		vertexSHProps.emplace_back(li + "r");
		vertexSHProps.emplace_back(li + "g");
		vertexSHProps.emplace_back(li + "b");
	}
	
	auto ply_vertices      = ply.request_properties_from_element("vertex", { "x", "y", "z" });
	auto ply_vertex_sh     = ply.request_properties_from_element("vertex", vertexSHProps);
	auto ply_tet_indices   = ply.request_properties_from_element("tetrahedron", { "vertex_indices" }, 4);
	auto ply_tet_densities = ply.request_properties_from_element("tetrahedron", { "density" });

	ply.read(file);

	auto getPlyData = []<typename T>(const auto& ply_data) -> std::span<T>{
		return std::span{reinterpret_cast<T*>(ply_data->buffer.get()), ply_data->buffer.size_bytes()/sizeof(T)};
	};

	std::span pos  = getPlyData.template operator()<float3>(ply_vertices);
	std::span sh   = getPlyData.template operator()<float3>(ply_vertex_sh);
	std::span inds = getPlyData.template operator()<uint4>(ply_tet_indices);
	std::span dens = getPlyData.template operator()<float>(ply_tet_densities);

	const uint32_t numTets = (uint32_t)inds.size();

	if (pos.size()*vertexSHDegree != sh.size()) {
		std::cerr << "Size of SH buffer " << sh.size() << " != " << pos.size()*vertexSHDegree << "!" << std::endl;
		return;
	}
	if (dens.size() != numTets) {
		std::cerr << "Size of density buffer " << dens.size() << " != " << numTets << "!" << std::endl;
		return;
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
	
	vertices = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer);
	vertexSH     = TexelBufferView::Create(context.GetDevice(), Buffer::Create(context.GetDevice(), sh.size()*sizeof(uint16_t)*3, vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eR16Sfloat);
	tetDensities = TexelBufferView::Create(context.GetDevice(), Buffer::Create(context.GetDevice(), numTets*sizeof(uint16_t),     vk::BufferUsageFlagBits::eUniformTexelBuffer|vk::BufferUsageFlagBits::eStorageBuffer), vk::Format::eR16Sfloat);
	tetIndices = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	tetCircumspheres = Buffer::Create(context.GetDevice(), numTets*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);
	
	// compress colors
	/*{
		Pipeline& pipeline = *compressColorsPipeline.get(context.GetDevice(), {
			{ "INPUT_TYPE",  "float3" },
			{ "OUTPUT_TYPE", "uint" },
			{ "COMPRESS_FN", "D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(i.bgr, 1))" },
		});
		ShaderParameter parameters;
		parameters["inputData"]  = (BufferParameter)context.UploadData(col, vk::BufferUsageFlagBits::eStorageBuffer);
		parameters["outputData"] = (BufferParameter)vertexColors.GetBuffer();
		parameters["count"] = (uint32_t)vertexColors.size();
		context.Dispatch(pipeline, (uint32_t)vertexColors.size(), parameters);
	}*/

	{
		Pipeline& f32tof16pipeline = *compressColorsPipeline.get(context.GetDevice(), {
			{ "INPUT_TYPE",  "float" },
			{ "OUTPUT_TYPE", "uint16_t" },
			{ "COMPRESS_FN", "(uint16_t)f32tof16(i)" },
		});
			
		// compress densities to float16
		{
			ShaderParameter parameters;
			parameters["inputData"]  = (BufferParameter)context.UploadData(dens, vk::BufferUsageFlagBits::eStorageBuffer);
			parameters["outputData"] = (BufferParameter)tetDensities.GetBuffer();
			parameters["count"] = numTets;
			context.Dispatch(f32tof16pipeline, numTets, parameters);
		}

		// compress SH coefficients to float16
		{
			const uint32_t n = (uint32_t)(sh.size()*3);
			ShaderParameter parameters;
			parameters["inputData"]  = (BufferParameter)context.UploadData(sh, vk::BufferUsageFlagBits::eStorageBuffer);
			parameters["outputData"] = (BufferParameter)vertexSH.GetBuffer();
			parameters["count"] = n;
			context.Dispatch(f32tof16pipeline, n, parameters);
		}
	}


	ComputeSpheres(context);
}

ShaderParameter TetrahedronScene::GetShaderParameter() {
	ShaderParameter sceneParams = {};
	sceneParams["vertices"]              = (BufferParameter)vertices;
	sceneParams["vertexSH"]              = (TexelBufferParameter)vertexSH;
	sceneParams["tetDensities"]          = (TexelBufferParameter)tetDensities;
	sceneParams["tetIndices"]            = (BufferParameter)tetIndices;
	sceneParams["tetCircumspheres"]      = (BufferParameter)tetCircumspheres;
	sceneParams["aabbMin"]        = minVertex;
	sceneParams["aabbMax"]        = maxVertex;
	sceneParams["densityScale"]   = densityScale;
	sceneParams["numTets"]        = TetCount();
	sceneParams["numVertices"]    = (uint32_t)vertices.size();
	sceneParams["vertexSHDegree"] = vertexSHDegree;
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
		totalSize += vertexSH        .size_bytes();
		totalSize += tetIndices      .size_bytes();   
		totalSize += tetDensities    .size_bytes();       
		totalSize += tetCircumspheres.size_bytes();
		const auto[x, unit] = FormatBytes(totalSize);
		ImGui::Text("%lu%s", x, unit);
	}

	ImGui::Text("SH degree: %u", vertexSHDegree);

	ImGui::Separator();
	ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
	ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
	ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
	ImGui::Separator();
	Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);
}