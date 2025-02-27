#include <portable-file-dialogs.h>
#include <tinyply.h>

#include "TetrahedronScene.hpp"

using namespace vkDelTet;

void TetrahedronScene::ComputeSpheres(CommandContext& context) {
	ShaderParameter parameters = {};
	parameters["scene"] = GetShaderParameter();
	parameters["outputSpheres"] = (BufferParameter)spheres;

	context.Dispatch(*createSpheresPipeline.get(context.GetDevice()), (uint32_t)indices.size(), parameters);
}

void TetrahedronScene::Load(CommandContext& context, const std::filesystem::path& p) {
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

	context.GetDevice().Wait(); // wait in case previous vertices/colors/indices are in use still

	vertices = context.UploadData(pos,  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eVertexBuffer);
	colors   = context.UploadData(col,  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eVertexBuffer);
	indices  = context.UploadData(inds, vk::BufferUsageFlagBits::eStorageBuffer);
	spheres = Buffer::Create(context.GetDevice(), numTets*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);

	maxDensity = 0;
	for (const float4 c : col)
		maxDensity = max(maxDensity, c.w);

	minVertex = float3( FLT_MAX );
	maxVertex = float3( FLT_MIN );
	for (float3 p : pos) {
		minVertex = min(p, minVertex);
		maxVertex = max(p, maxVertex);
	}

	ComputeSpheres(context);
}

ShaderParameter TetrahedronScene::GetShaderParameter() {
	ShaderParameter sceneParams = {};
	sceneParams["vertices"] = (BufferParameter)vertices;
	sceneParams["colors"]   = (BufferParameter)colors;
	sceneParams["indices"]  = (BufferParameter)indices;
	sceneParams["spheres"]  = (BufferParameter)spheres;
	sceneParams["numTets"]  = (uint32_t)indices.size();
	sceneParams["aabbMin"] = minVertex;
	sceneParams["aabbMax"] = maxVertex;
	sceneParams["densityScale"] = densityScale;
	return sceneParams;
}

void TetrahedronScene::DrawGui(CommandContext& context) {
	ImGui::Text("%llu tetrahedra", indices.size());
	ImGui::Text("%llu vertices", vertices.size());
	ImGui::Separator();
	ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
	ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
	ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
	ImGui::Separator();
	Gui::ScalarField("Density scale", &densityScale, 0.f, 1e4f, 0.01f);
}