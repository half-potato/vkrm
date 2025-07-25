#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <map>

// Add the cxxopts header
#include "cxxopts.h"

#include "DelaunayTetRenderer.hpp"
#include "ColmapUtils.h"

using namespace vkDelTet;

int main(int argc, const char** argv) {
    // --- Argument Parsing ---
    cxxopts::Options options("TetRenderer", "A Delaunay tetrahedral mesh renderer benchmark tool.");
    options.add_options()
        ("s,scene", "Path to the scene file to render", cxxopts::value<std::string>())
        ("c,colmap", "Path to the COLMAP sparse reconstruction directory", cxxopts::value<std::string>())
        ("f,fov", "Use fovX/fovY instead of a single fov value (true/false)", cxxopts::value<bool>()->default_value("true"))
        ("l,llff_hold", "Take every Nth image for benchmark", cxxopts::value<int>()->default_value("8"))
        ("d,downsample", "Downsample factor for image resolution", cxxopts::value<int>()->default_value("4"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("scene") || !result.count("colmap")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string scenePath = result["scene"].as<std::string>();
    const std::string colmapSparsePath = result["colmap"].as<std::string>();
    const bool fovXfovYFlag = result["fov"].as<bool>();
    const int llffHold = result["llff_hold"].as<int>();
    const int downsampleFactor = result["downsample"].as<int>();
    // -------------------------

    WindowedApp app("TetRenderer", {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME
    });

    // --- Camera Loading and Filtering ---
    auto allCamerasMap = loadColmapBin(colmapSparsePath, 0.2f, fovXfovYFlag);
	// TransformPosesPCA(allCamerasMap);

    // Convert map to a vector to have a stable, indexable order.
    std::vector<ColmapCamera> allCamerasVec;
    allCamerasVec.reserve(allCamerasMap.size());
    for (auto const& [key, val] : allCamerasMap) {
        allCamerasVec.push_back(val);
    }

    // Apply the llff_hold filter to get the final list of cameras for the benchmark.
    std::vector<ColmapCamera> benchmarkCameras;
    if (llffHold > 0) {
        for (int i = 0; i < allCamerasVec.size(); ++i) {
            if (i % llffHold == 0) {
                benchmarkCameras.push_back(allCamerasVec[i]);
            }
        }
    } else {
        benchmarkCameras = allCamerasVec; // Use all cameras if hold is 0 or less.
    }
    std::cout << "Loaded " << allCamerasMap.size() << " cameras, benchmarking with " << benchmarkCameras.size() << " cameras." << std::endl;
    // ------------------------------------

    DelaunayTetRenderer renderer;

    app.contexts[0]->Begin();
    renderer.LoadScene(*app.contexts[0], scenePath);

    renderer.renderContext.scene.sceneTranslation = float3(0);
    renderer.renderContext.scene.sceneRotation = float3(0, 0, 0);
    app.contexts[0]->Submit();
    
    // --- Benchmark State Variables ---
    bool isBenchmarking = false;
    int currentCameraIndex = 0;
    int frameCount = 0;
    double benchmarkDuration = 0.5; // Seconds per camera
    std::chrono::steady_clock::time_point startTime;
    std::vector<float> fpsResults;
    // ---------------------------------

    app.AddWidget("Viewport", [&]() {
        // --- Benchmark Start Trigger ---
        if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiMod_Ctrl) && !isBenchmarking) {
			printf("Starting benchmark...\n");
            if (!benchmarkCameras.empty()) {
                isBenchmarking = true;
                currentCameraIndex = 0;
                frameCount = 0;
                fpsResults.clear();

                // Start with the first camera
                auto& camData = benchmarkCameras[currentCameraIndex];
                renderer.renderContext.camera = camData.camera;
                // Apply downsampling to the resolution
                renderer.renderContext.overrideResolution = camData.dimensions / (uint)downsampleFactor;
                
                startTime = std::chrono::steady_clock::now();
                std::cout << "Starting benchmark for camera " << currentCameraIndex 
                          << " at " << renderer.renderContext.overrideResolution->x 
                          << "x" << renderer.renderContext.overrideResolution->y << std::endl;
            }
        }

        // --- Per-Frame Benchmark Logic ---
        if (isBenchmarking) {
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();

            if (elapsedTime >= benchmarkDuration) {
                float fps = static_cast<float>(frameCount) / elapsedTime;
                fpsResults.push_back(fps);
                std::cout << "  -> Result: " << fps << " FPS" << std::endl;

                // Move to the next camera
                currentCameraIndex++;
                if (currentCameraIndex < benchmarkCameras.size()) {
                    frameCount = 0;
                    auto& camData = benchmarkCameras[currentCameraIndex];
                    renderer.renderContext.camera = camData.camera;
                    // Apply downsampling to the next camera's resolution
                    renderer.renderContext.overrideResolution = camData.dimensions / (uint)downsampleFactor;
                    
                    startTime = std::chrono::steady_clock::now();
                    std::cout << "Starting benchmark for camera " << currentCameraIndex 
                              << " at " << renderer.renderContext.overrideResolution->x 
                              << "x" << renderer.renderContext.overrideResolution->y << std::endl;
                } else {
                    // Benchmark finished
                    isBenchmarking = false;
                    renderer.renderContext.overrideResolution.reset(); // Revert to UI-driven resolution
                    std::cout << "Benchmark finished!" << std::endl;
                    // Optional: Print average FPS
                    float totalFps = 0.0f;
                    for(float fps : fpsResults) totalFps += fps;
                    std::cout << "Average FPS: " << totalFps / fpsResults.size() << std::endl;
                }
            }
        }

        renderer.DrawWidgetGui(*app.contexts[app.swapchain->ImageIndex()], app.dt);
    }, true, WindowedApp::WidgetFlagBits::eNoBorders);

    app.Run();
    app.device->Wait();
    return EXIT_SUCCESS;
}

