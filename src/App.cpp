#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>

#include "DelaunayTetRenderer.hpp"

using namespace RoseEngine;

int main(int argc, const char** argv) {
	WindowedApp app("DelaunayTetRenderer", {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	});

	DelaunayTetRenderer renderer;

	if (argc > 1) {
		app.contexts[0]->Begin();
		renderer.LoadScene(*app.contexts[0], argv[1]);
		app.contexts[0]->Submit();
	}

	app.AddMenuItem("File", [&]() {
		if (ImGui::MenuItem("Open scene")) {
			auto f = pfd::open_file(
				"Choose scene",
				"",
				{ "PLY files (.ply)", "*.ply" },
				false
			);
			for (const std::string& filepath : f.result()) {
				renderer.LoadScene(*app.contexts[app.swapchain->ImageIndex()], filepath);
			}
		}
	});

	app.AddWidget("Properties", [&]() {
		renderer.RenderProperties(*app.contexts[app.swapchain->ImageIndex()]);
	}, true);

	app.AddWidget("Viewport", [&]() {
		renderer.RenderWidget(*app.contexts[app.swapchain->ImageIndex()], app.dt);
	}, true);

	app.Run();

	app.device->Wait();

	return EXIT_SUCCESS;
}
