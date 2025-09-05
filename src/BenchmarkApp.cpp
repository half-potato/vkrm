#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <vulkan/vulkan_enums.hpp>

// Add the cxxopts header
#include "cxxopts.h"

#include "DelaunayTetRenderer.hpp"
#include "ColmapUtils.h"

using namespace vkDelTet;

Eigen::Matrix4f loadMatrixFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Error: Could not open the file " + filename);
    }

    std::vector<float> data;
    float number;
    while (inputFile >> number) {
        data.push_back(number);
    }

    if (data.size() < 16) {
        throw std::runtime_error("Error: File does not contain enough data for a 4x4 matrix.");
    }

    // Map the vector data to a 4x4 row-major matrix and return it.
    return Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(data.data());
}

int main(int argc, const char** argv) {
    // --- Argument Parsing ---
    cxxopts::Options options("TetRenderer", "A Delaunay tetrahedral mesh renderer benchmark tool.");
    options.add_options()
        ("s,scene", "Path to the scene file to render", cxxopts::value<std::string>())
        ("c,colmap", "Path to the COLMAP sparse reconstruction directory", cxxopts::value<std::string>())
        ("f,fov", "Use fovX/fovY instead of a single fov value (true/false)", cxxopts::value<bool>()->default_value("true"))
        ("t,transform_file", "Path to a 4x4 transform matrix file", cxxopts::value<std::string>())
        ("n,no_pca", "Disable pca for camera positions(true/false)", cxxopts::value<bool>()->default_value("false"))
        ("a,auto", "Auto startup and shutdown", cxxopts::value<bool>()->default_value("false"))
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
    app.swapchain->SetPresentMode(vk::PresentModeKHR::eImmediate);

    // --- Camera Loading and Filtering ---
    auto allCamerasMap = loadColmapBin(colmapSparsePath, 0.2f, fovXfovYFlag);
    // Priority: 1. Use transform_file, 2. Honor no_pca, 3. Default to PCA
    if (result.count("transform_file"))
    {
        // 1. If a transform file is provided, load and apply it.
        const std::string transformPath = result["transform_file"].as<std::string>();
        std::cout << "Loading transformation from file: " << transformPath << std::endl;
        Eigen::Matrix4f transform = loadMatrixFromFile(transformPath);
        TransformCameras(allCamerasMap, transform);
    }
    else if (result["no_pca"].as<bool>())
    {
        // 2. If no_pca is true, do nothing.
        std::cout << "Skipping PCA transformation." << std::endl;
    }
    else
    {
        // 3. Otherwise, default to calculating and applying PCA.
        std::cout << "Calculating PCA transformation." << std::endl;
        Eigen::Matrix4f transform = PosesPCA(allCamerasMap);
        TransformCameras(allCamerasMap, transform);
    }

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
    std::chrono::steady_clock::time_point intervalTime;
    std::vector<float> fpsResults;

    app.AddMenuItem("File", [&]() {
        if (ImGui::MenuItem("Open scene")) {
            openSceneDialog();
        }
    });

    app.AddWidget("Properties", [&]() {
        renderer.DrawPropertiesGui(*app.contexts[app.swapchain->ImageIndex()]);
    }, true);

    // ---------------------------------
    auto& camData = benchmarkCameras[0];
    renderer.renderContext.camera = camData.camera;

    app.AddWidget("Viewport", [&]() {
        bool start = (renderer.frame_count > 500 and result["auto"].as<bool>() and not isBenchmarking);
        // --- Benchmark Start Trigger ---
        start |= (ImGui::IsKeyPressed(ImGuiKey_O) and ImGui::IsKeyDown(ImGuiMod_Ctrl) and !isBenchmarking);
        if (start) {
            printf("Starting benchmark...\n");
            if (!benchmarkCameras.empty()) {
                isBenchmarking = true;
                currentCameraIndex = 0;
                frameCount = 0;
                fpsResults.clear();

                // Start with the first camera
                auto& camData = benchmarkCameras[currentCameraIndex];
                renderer.renderContext.camera = camData.camera;
                renderer.renderContext.overrideResolution = camData.dimensions / (uint)downsampleFactor;

                startTime = std::chrono::steady_clock::now();
                intervalTime = std::chrono::steady_clock::now();
                std::cout << "Starting benchmark for camera " << currentCameraIndex 
                    << " at " << renderer.renderContext.overrideResolution->x 
                    << "x" << renderer.renderContext.overrideResolution->y << std::endl;
            }
        }

        // --- Per-Frame Benchmark Logic ---
        if (isBenchmarking) {
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - intervalTime).count();

            if (elapsedTime >= benchmarkDuration) {
                // float fps = static_cast<float>(frameCount) / elapsedTime;
                // fpsResults.push_back(fps);
                // std::cout << "  -> Result: " << fps << " FPS" << std::endl;

                // Move to the next camera
                currentCameraIndex++;
                if (currentCameraIndex < benchmarkCameras.size()) {
                    auto& camData = benchmarkCameras[currentCameraIndex];
                    renderer.renderContext.camera = camData.camera;
                    // Apply downsampling to the next camera's resolution
                    renderer.renderContext.overrideResolution = camData.dimensions / (uint)downsampleFactor;

                    // startTime = std::chrono::steady_clock::now();
                    intervalTime = std::chrono::steady_clock::now();
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
                    // for(float fps : fpsResults) totalFps += fps;
                    // std::cout << "Average FPS: " << totalFps / fpsResults.size() << std::endl;
                    double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
                    float fps = static_cast<float>(frameCount) / elapsedTime;
                    std::cout << "Average FPS: " << fps << std::endl;
                    Gui::Destroy();
                    exit(0);
                }
            }
        }

        renderer.DrawWidgetGui(*app.contexts[app.swapchain->ImageIndex()], app.dt);
    }, true, WindowedApp::WidgetFlagBits::eNoBorders);

    app.Run();
    app.device->Wait();
    return EXIT_SUCCESS;
}

