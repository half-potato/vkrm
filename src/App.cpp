#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>

#include "DelaunayTetRenderer.hpp"

using namespace vkDelTet;

int main(int argc, const char** argv) {
	WindowedApp app("TetRenderer", {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	});

	DelaunayTetRenderer renderer;

	auto openSceneDialog = [&]() {
		auto f = pfd::open_file(
			"Choose scene",
			"",
			{ "PLY files (.ply)", "*.ply" },
			false
		);
		for (const std::string& filepath : f.result()) {
			renderer.LoadScene(*app.contexts[app.swapchain->ImageIndex()], filepath);
		}
	};

	if (argc > 1) {
		app.contexts[0]->Begin();
		renderer.LoadScene(*app.contexts[0], argv[1]);
		app.contexts[0]->Submit();
	}

	app.AddMenuItem("File", [&]() {
		if (ImGui::MenuItem("Open scene")) {
			openSceneDialog();
		}
	});

	app.AddWidget("Properties", [&]() {
		renderer.DrawPropertiesGui(*app.contexts[app.swapchain->ImageIndex()]);
	}, true);

	app.AddWidget("Viewport", [&]() {
		if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiMod_Ctrl)) {
			openSceneDialog();
		}
		renderer.DrawWidgetGui(*app.contexts[app.swapchain->ImageIndex()], app.dt);
	}, true, WindowedApp::WidgetFlagBits::eNoBorders);

	app.Run();

	app.device->Wait();

	return EXIT_SUCCESS;
}
