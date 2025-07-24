#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>

#include <Rose/Render/ViewportCamera.hpp> // Your ViewportCamera header
#include <glm/gtc/quaternion.hpp>         // For quat_cast, mat3_cast
#include <Eigen/Dense>
#include <glm/gtc/type_ptr.hpp>


// --- Type Aliases ---
using namespace RoseEngine;
using float3 = glm::vec3;
using quat = glm::quat;
using mat3 = glm::mat3;
using uint2 = glm::uvec2;

// Struct to hold both the camera and its original dimensions
struct ColmapCamera {
    ViewportCamera camera;
    uint2 dimensions;
};

// --- Helper functions from your code ---
// (IsLittleEndian, ReverseBytes, LittleEndianToNative, ReadBinaryLittleEndian)
inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
    return false;
#else
    return true;
#endif
}

template <typename T>
T ReverseBytes(T x) {
    T reversed_x;
    char* src = reinterpret_cast<char*>(&x);
    char* dst = reinterpret_cast<char*>(&reversed_x);
    for (size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = src[sizeof(T) - 1 - i];
    }
    return reversed_x;
}

template <typename T>
T LittleEndianToNative(const T x) {
    if (IsLittleEndian()) {
        return x;
    } else {
        return ReverseBytes(x);
    }
}

template <typename T>
T ReadBinaryLittleEndian(std::istream* stream) {
    T data_little_endian;
    stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
    return LittleEndianToNative(data_little_endian);
}

inline glm::mat4 TransformPosesPCA(std::map<std::string, ColmapCamera>& cameras) {
    if (cameras.empty()) {
        return glm::mat4(1.0f);
    }

    // 1. Extract all camera positions (translations) into an Eigen matrix
    Eigen::MatrixXf positions(cameras.size(), 3);
    int i = 0;
    for (const auto& pair : cameras) {
        const auto& p = pair.second.camera.position;
        positions.row(i++) << p.x, p.y, p.z;
    }

    // 2. Compute the mean and center the positions
    Eigen::Vector3f t_mean = positions.colwise().mean();
    positions.rowwise() -= t_mean.transpose();

    // 3. Compute the covariance matrix and its eigendecomposition for PCA
    Eigen::Matrix3f covariance = positions.transpose() * positions;
    Eigen::EigenSolver<Eigen::Matrix3f> es(covariance);
    Eigen::Matrix3f eigvec = es.eigenvectors().real();

    // The eigenvalues are not guaranteed to be sorted, but Eigen's solver often
    // returns them in a reasonable order. For perfect correspondence with the python
    // script, one would sort eigenvectors by eigenvalues. Here we assume the order is sufficient.

    // 4. Create the rotation matrix from the eigenvectors
    Eigen::Matrix3f rot = eigvec.transpose();

    // 5. Ensure a right-handed coordinate system
    if (rot.determinant() < 0) {
        rot.row(2) *= -1;
    }

    // 6. Construct the final 4x4 transformation matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rot;
    transform.block<3, 1>(0, 3) = rot * -t_mean;

    // 7. Apply the transformation to all camera poses
    for (auto& pair : cameras) {
        // Get the original pose as a 4x4 matrix
        glm::mat4 original_pose = glm::translate(glm::mat4(1.0f), pair.second.camera.position);
        original_pose = original_pose * glm::mat4_cast(pair.second.camera.GetRotation());

        // Convert to Eigen, apply transform, and convert back
        Eigen::Matrix4f original_pose_eigen;
        memcpy(original_pose_eigen.data(), glm::value_ptr(original_pose), sizeof(float) * 16);

        Eigen::Matrix4f new_pose_eigen = transform * original_pose_eigen;
        
        glm::mat4 new_pose_glm;
        memcpy(glm::value_ptr(new_pose_glm), new_pose_eigen.data(), sizeof(float) * 16);

        // Decompose the new matrix back into position and rotation
        pair.second.camera.position = glm::vec3(new_pose_glm[3]);
        pair.second.camera.rotation = glm::quat_cast(new_pose_glm);
    }

    // The final axis flip check from the python script
    // This heuristic orients the scene to be upright
    float3 avg_z = float3(0.0f);
    for (const auto& pair : cameras) {
        avg_z += pair.second.camera.GetRotation() * float3(0,0,1);
    }
    avg_z = glm::normalize(avg_z);

    if (glm::dot(avg_z, float3(0,1,0)) < 0) {
        glm::mat4 flip = glm::scale(glm::mat4(1.0f), glm::vec3(1, -1, -1));
        
        // Update the transform matrix
        Eigen::Matrix4f flip_eigen;
        memcpy(flip_eigen.data(), glm::value_ptr(flip), sizeof(float) * 16);
        transform = flip_eigen * transform;

        // Re-apply the flip to all poses
        for (auto& pair : cameras) {
             pair.second.camera.position = flip * glm::vec4(pair.second.camera.position, 1.0f);
             pair.second.camera.rotation = glm::quat_cast(flip) * pair.second.camera.rotation;
        }
    }


    // Convert final Eigen transform to GLM for the return value
    glm::mat4 final_transform_glm;
    memcpy(glm::value_ptr(final_transform_glm), transform.data(), sizeof(float) * 16);

    return final_transform_glm;
}


// --- Modified Function (GLM Only) ---

// std::map<std::string, ColmapCamera> loadColmapBin(const std::string& colmapSparsePath, const float zNear, const int fovXfovYFlag)
std::map<std::string, ColmapCamera> loadColmapBin(const std::string& colmapSparsePath, const float zNear, const int fovXfovYFlag)
{
    const std::string camerasListing = colmapSparsePath + "/cameras.bin";
    const std::string imagesListing  = colmapSparsePath + "/images.bin";

    std::ifstream camerasFile(camerasListing, std::ios::binary);
    std::ifstream imagesFile(imagesListing, std::ios::binary);

    if (!camerasFile.is_open()) {
        std::cerr << "ERROR: Unable to load camera colmap file: " << camerasListing << std::endl;
        return {};
    }
    if (!imagesFile.is_open()) {
        std::cerr << "ERROR: Unable to load images colmap file: " << imagesListing << std::endl;
        return {};
    }

    std::map<std::string, ColmapCamera> cameras;

    struct CameraParametersColmap {
        uint32_t id;
        uint64_t width;
        uint64_t height;
        double   fx, fy, dx, dy;
    };

    std::map<uint32_t, CameraParametersColmap> cameraParameters;
    const uint64_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&camerasFile);

    for (size_t i = 0; i < num_cameras; ++i) {
        CameraParametersColmap params;
        params.id = ReadBinaryLittleEndian<uint32_t>(&camerasFile);
        int model_id = ReadBinaryLittleEndian<int>(&camerasFile);
        params.width = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
        params.height = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
        
        params.fx = ReadBinaryLittleEndian<double>(&camerasFile);
        params.fy = ReadBinaryLittleEndian<double>(&camerasFile);
        params.dx = ReadBinaryLittleEndian<double>(&camerasFile);
        params.dy = ReadBinaryLittleEndian<double>(&camerasFile);
        
        cameraParameters[params.id] = params;
    }

    // Coordinate system conversion matrix using GLM
    Eigen::Matrix3f converter;
    converter << 1, 0,  0,
                 0, -1, 0,
                 0, 0, -1;

    const uint64_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
    for (size_t i = 0; i < num_reg_images; ++i) {
        uint32_t image_id = ReadBinaryLittleEndian<uint32_t>(&imagesFile);
        Eigen::Quaternionf q_colmap;
        q_colmap.w() = ReadBinaryLittleEndian<double>(&imagesFile);
        q_colmap.x() = ReadBinaryLittleEndian<double>(&imagesFile);
        q_colmap.y() = ReadBinaryLittleEndian<double>(&imagesFile);
        q_colmap.z() = ReadBinaryLittleEndian<double>(&imagesFile);
        
        Eigen::Vector3f t_colmap;
        t_colmap.x() = ReadBinaryLittleEndian<double>(&imagesFile);
        t_colmap.y() = ReadBinaryLittleEndian<double>(&imagesFile);
        t_colmap.z() = ReadBinaryLittleEndian<double>(&imagesFile);
        
        uint32_t camera_id = ReadBinaryLittleEndian<uint32_t>(&imagesFile);
        
        std::string image_name;
        char name_char;
        do {
            imagesFile.read(&name_char, 1);
            if (name_char != '\0') {
                image_name += name_char;
            }
        } while (name_char != '\0');

        if (cameraParameters.find(camera_id) == cameraParameters.end()) {
            std::cerr << "WARNING: Could not find intrinsics for camera_id: " << camera_id << ". Skipping image " << image_name << std::endl;
            continue;
        }
        const CameraParametersColmap& camParams = cameraParameters.at(camera_id);

        Eigen::Matrix3f R_colmap = q_colmap.toRotationMatrix();
        Eigen::Matrix3f R_graphics = R_colmap * converter;
        Eigen::Vector3f p_graphics = R_colmap * t_colmap;

        glm::quat final_rotation = glm::make_quat(R_graphics.data());
        glm::vec3 final_position = glm::make_vec3(p_graphics.data());

        const float fovY_rad = 2.0f * atanf(camParams.height / (2.0f * camParams.fy));
        const float fovX_rad = 2.0f * atanf(camParams.width / (2.0f * camParams.fx));
        const float fovY_deg = glm::degrees(fovY_rad);
        const float fovX_deg = glm::degrees(fovX_rad);
        
        // Create the ViewportCamera as before
        ViewportCamera viewport_cam(final_position, final_rotation, fovX_deg, fovY_deg, zNear);
        viewport_cam.projectionMode = (fovXfovYFlag)
            ? ViewportCamera::ProjectionMode::FovXY
            : ViewportCamera::ProjectionMode::FovY;

        ColmapCamera final_cam_data;
        final_cam_data.camera = viewport_cam;
        final_cam_data.dimensions = uint2(camParams.width, camParams.height);

        cameras[image_name] = final_cam_data;

        // Ignore the 2D points data
        const uint64_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
        imagesFile.seekg(num_points2D * (sizeof(double) * 2 + sizeof(uint64_t)), std::ios_base::cur);
    }
    
    return cameras;
}
