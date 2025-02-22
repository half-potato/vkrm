#pragma once

#include <Rose/Core/Gui.hpp>

namespace RoseEngine {

struct Camera {
	float3 position = float3(0, 2, 4);
	float2 eulerAngles = float2(-float(M_PI)/4, 0);
	float  fovY = glm::radians(50.f);
	float  nearZ = 0.01f;
	float  farZ = 1000.0;

	float moveSpeed = 1.f;

	inline quat Rotation() const {
		quat rx = glm::angleAxis(eulerAngles.x, float3(1,0,0));
		quat ry = glm::angleAxis(eulerAngles.y, float3(0,1,0));
		return ry * rx;
	}

	inline void Update(double dt) {
		if (ImGui::IsWindowHovered()) {
			if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
				eulerAngles += -float2(ImGui::GetIO().MouseDelta.y, ImGui::GetIO().MouseDelta.x) * float(M_PI) / 1920.f;
				eulerAngles.x = clamp(eulerAngles.x, -float(M_PI/2), float(M_PI/2));
			}
		}

		if (ImGui::IsWindowFocused()) {
			if (ImGui::GetIO().MouseWheel != 0) {
				moveSpeed *= (1 + ImGui::GetIO().MouseWheel / 8);
				moveSpeed = std::max(moveSpeed, .05f);
			}

			float3 move = float3(0,0,0);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_W)) move += float3( 0, 0,-1);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_S)) move += float3( 0, 0, 1);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_D)) move += float3( 1, 0, 0);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_A)) move += float3(-1, 0, 0);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_Q)) move += float3( 0,-1, 0);
			if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_E)) move += float3( 0, 1, 0);
			if (move != float3(0,0,0)) {
				move = Rotation() * normalize(move);
				if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
					move *= 3.f;
                position += move * moveSpeed * float(dt);
			}
		}
	}

    inline void DrawGui() {
        ImGui::PushID("Camera");

		ImGui::DragFloat3("Position", &position.x);
		ImGui::DragFloat("Pitch", &eulerAngles.x, float(M_1_PI), -float(M_PI/2), float(M_PI/2));
		ImGui::DragFloat("Yaw",   &eulerAngles.y, float(M_1_PI), -float(M_PI), float(M_PI));
		ImGui::DragFloat("Near Z", &nearZ, 0.01f, 1e-6f);
		ImGui::DragFloat("Far Z", &nearZ, 0.01f, nearZ);
        ImGui::DragFloat("Vertical FoV", &fovY, float(M_1_PI), 0.f, (float)M_PI);
        ImGui::DragFloat("Move speed", &moveSpeed, 1.f, 0.f, 1e9f);

        ImGui::PopID();
    }

    inline float4x4 GetCameraToWorld() {
        return glm::translate(position) * (float4x4)Rotation();
    }

    // aspect = x/y
    inline float4x4 GetProjection(float aspect) {
        float4x4 p = glm::perspective(fovY, aspect, nearZ, farZ);
		p[1][1] = -p[1][1]; // vulkan is y-down, vertically flip the projection
		return p;
    }
};

}