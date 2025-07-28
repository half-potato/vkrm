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
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covariance);

    // Sort eigenvectors by eigenvalues in descending order to match np.argsort
    std::vector<std::pair<float, Eigen::Vector3f>> eigen_pairs;
    for (int j = 0; j < 3; ++j) {
        eigen_pairs.push_back({es.eigenvalues()[j], es.eigenvectors().col(j)});
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    Eigen::Matrix3f eigvec;
    for (int j = 0; j < 3; ++j) {
        eigvec.col(j) = eigen_pairs[j].second;
    }

    // 4. Create the rotation matrix from the sorted eigenvectors
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
        glm::mat4 original_pose = glm::translate(glm::mat4(1.0f), pair.second.camera.position) * glm::mat4_cast(pair.second.camera.GetRotation());

        Eigen::Map<const Eigen::Matrix4f> original_pose_eigen(glm::value_ptr(original_pose));
        Eigen::Matrix4f new_pose_eigen = transform * original_pose_eigen;
        
        glm::mat4 new_pose_glm;
        memcpy(glm::value_ptr(new_pose_glm), new_pose_eigen.data(), sizeof(float) * 16);

        pair.second.camera.position = glm::vec3(new_pose_glm[3]);
        pair.second.camera.rotation = glm::quat_cast(new_pose_glm);
    }

    // --- CORRECTED FLIP HEURISTIC ---
    // This now exactly matches the Python script's logic: poses_recentered.mean(axis=0)[2, 1] < 0
    float avg_y_axis_z_comp = 0.0f;
    for (const auto& pair : cameras) {
        // Get the rotation matrix of the new pose
        glm::mat3 r = glm::mat3_cast(pair.second.camera.GetRotation());
        // Get the element at row 2, column 1 (y_z component)
        avg_y_axis_z_comp += r[1][2];
    }
    avg_y_axis_z_comp /= cameras.size();

    // if (avg_y_axis_z_comp < 0) {
    //     glm::mat4 flip = glm::scale(glm::mat4(1.0f), glm::vec3(1, -1, -1));
    //     
    //     Eigen::Map<const Eigen::Matrix4f> flip_eigen(glm::value_ptr(flip));
    //     transform = flip_eigen * transform;
    //
    //     // Re-apply the flip to all poses
    //     for (auto& pair : cameras) {
    //          pair.second.camera.position = flip * glm::vec4(pair.second.camera.position, 1.0f);
    //          pair.second.camera.rotation = glm::quat_cast(flip) * pair.second.camera.rotation;
    //     }
    // }
    // --- END CORRECTION ---

    // Convert final Eigen transform to GLM for the return value
    glm::mat4 final_transform_glm;
    memcpy(glm::value_ptr(final_transform_glm), transform.data(), sizeof(float) * 16);

    return final_transform_glm;
}


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

    const uint64_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
    for (size_t i = 0; i < num_reg_images; ++i) {
        uint32_t image_id = ReadBinaryLittleEndian<uint32_t>(&imagesFile);
        
        // --- THIS IS THE CORRECTED LOGIC ---

        // 1. Read Colmap's World-to-Camera (W2C) transform
        quat q_w2c;
        q_w2c.w = ReadBinaryLittleEndian<double>(&imagesFile);
        q_w2c.x = ReadBinaryLittleEndian<double>(&imagesFile);
        q_w2c.y = ReadBinaryLittleEndian<double>(&imagesFile);
        q_w2c.z = ReadBinaryLittleEndian<double>(&imagesFile);

        float3 t_w2c;
        t_w2c.x = ReadBinaryLittleEndian<double>(&imagesFile);
        t_w2c.y = ReadBinaryLittleEndian<double>(&imagesFile);
        t_w2c.z = ReadBinaryLittleEndian<double>(&imagesFile);

        printf("xyzw: %f, %f, %f, %f, t: %f, %f, %f\n", q_w2c.x, q_w2c.y, q_w2c.z, q_w2c.w, t_w2c.x, t_w2c.y, t_w2c.z);

        // 2. Invert the W2C transform to get the Camera-to-World (C2W) pose
        mat3 R_w2c = glm::mat3_cast(q_w2c);
        mat3 R_c2w = glm::transpose(R_w2c);
        float3 t_c2w = -R_c2w * t_w2c;

        // 3. Apply coordinate system change from Colmap (Y down, Z forward) to Graphics (Y up, Z back)
        // This is equivalent to right-multiplying the pose by a diagonal matrix with [1, -1, -1]
        // t_c2w.y *= -1.0f;
        // t_c2w.z *= -1.0f;
        R_c2w[1] *= -1.0f; // Flip the Y column
        R_c2w[2] *= -1.0f; // Flip the Z column
        
        quat final_rotation = glm::quat_cast(R_c2w);
        float3 final_position = t_c2w;
        printf("f xyzw: %f, %f, %f, %f, t: %f, %f, %f\n", final_rotation.x, final_rotation.y, final_rotation.z, final_rotation.w, final_position.x, final_position.y, final_position.z);

        // --- END OF CORRECTED LOGIC ---
        
        uint32_t camera_id = ReadBinaryLittleEndian<uint32_t>(&imagesFile);
        
        std::string image_name;
        char name_char;
        do {
            imagesFile.read(&name_char, 1);
            if (name_char != '\0') image_name += name_char;
        } while (name_char != '\0');

        if (cameraParameters.find(camera_id) == cameraParameters.end()) {
            std::cerr << "WARNING: Could not find intrinsics for camera_id: " << camera_id << ". Skipping image " << image_name << std::endl;
            continue;
        }
        const CameraParametersColmap& camParams = cameraParameters.at(camera_id);

        const float fovY_rad = 2.0f * atanf(camParams.height / (2.0f * camParams.fy));
        const float fovX_rad = 2.0f * atanf(camParams.width / (2.0f * camParams.fx));
        const float fovY_deg = glm::degrees(fovY_rad);
        const float fovX_deg = glm::degrees(fovX_rad);
        
        ViewportCamera viewport_cam(final_position, final_rotation, fovX_deg, fovY_deg, zNear);
        viewport_cam.projectionMode = (fovXfovYFlag)
            ? ViewportCamera::ProjectionMode::FovXY
            : ViewportCamera::ProjectionMode::FovY;

        ColmapCamera final_cam_data;
        final_cam_data.camera = viewport_cam;
        final_cam_data.dimensions = uint2(camParams.width, camParams.height);

        cameras[image_name] = final_cam_data;

        const uint64_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
        imagesFile.seekg(num_points2D * (sizeof(double) * 2 + sizeof(uint64_t)), std::ios_base::cur);
    }
    
    return cameras;
}
