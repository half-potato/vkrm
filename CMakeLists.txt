cmake_minimum_required(VERSION 3.20)

project(rmvk)

# We still find Vulkan for the *loader* (linking), but the headers 
# will effectively come from RoseLib's frozen version.
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(ROSE_ENABLE_TESTING OFF CACHE BOOL "")
set(ROSE_BUILD_APPS      OFF CACHE BOOL "")
add_subdirectory(extern/Rose)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src" SHADER_INCLUDE_DIR)
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/Rose/src/Rose/Core/ShaderModule.cpp" PROPERTIES COMPILE_FLAGS
    "-DDEFAULT_SHADER_INCLUDE_PATHS=\\\"${SHADER_INCLUDE_DIR}\\\"," )
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(rmvk
    src/App.cpp
    src/Scene/TetrahedronScene.cpp
)
set_target_properties(rmvk PROPERTIES LINKER_LANGUAGE CXX)

target_link_directories(rmvk PRIVATE /usr/local/lib)

# Link RoseLib FIRST. This ensures the frozen include directories in RoseLib
# take precedence over system Vulkan includes if there is a conflict.
target_link_libraries(rmvk PRIVATE RoseLib Eigen3::Eigen)
target_link_libraries(rmvk PUBLIC Vulkan::Vulkan glm)

target_compile_definitions(rmvk PUBLIC WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES GLM_FORCE_XYZW_ONLY IMGUI_DEFINE_MATH_OPERATORS VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

add_executable(benchmark
    src/BenchmarkApp.cpp
    src/Scene/TetrahedronScene.cpp
)
set_target_properties(benchmark PROPERTIES LINKER_LANGUAGE CXX)

target_link_directories(benchmark PRIVATE /usr/local/lib)

# Same here: RoseLib first
target_link_libraries(benchmark PRIVATE RoseLib Eigen3::Eigen)
target_link_libraries(benchmark PUBLIC Vulkan::Vulkan glm)

target_compile_definitions(benchmark PUBLIC WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES GLM_FORCE_XYZW_ONLY IMGUI_DEFINE_MATH_OPERATORS VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)
