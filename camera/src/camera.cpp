#include <opencv2/core/persistence.hpp>

#include <camera.h>

void Camera::read_config() {
    std::string config_file =
        std::string(PACKAGE_ROOT_DIR) + "/config/config.yml";
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(config_file);
    }

    camera_id = static_cast<int>(fs["camera_id"]);
}

void Camera::read_calibration_result() {
    std::string calib_result_path =
        std::string(PACKAGE_ROOT_DIR) + "/config/calibration_result.yml";
    cv::FileStorage fs(calib_result_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw FileNotFoundException(calib_result_path);
    }
    fs["cameraMatrix"] >> calib_res.camera_matrix;
    fs["distCoeffs"] >> calib_res.dist_coeffs;
}

void Camera::create_camera(int camera_id) {
    cap = cv::VideoCapture(camera_id);
    if (!cap.isOpened()) {
        throw CameraOpenException(camera_id);
    }
}

cv::Mat Camera::read_frame() {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        throw FrameCaptureException();
    }

    cv::Mat undist_frame;
    cv::undistort(frame, undist_frame, calib_res.camera_matrix,
                  calib_res.dist_coeffs);

    return undist_frame;
}

Camera::Camera() {
    read_config();
    read_calibration_result();
    create_camera(camera_id);
}
