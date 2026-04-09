#pragma once

#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

struct CalibrationResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

class Camera {
private:
    int camera_id;
    cv::VideoCapture cap;
    CalibrationResult calib_res;

    void read_config();
    void read_calibration_result();
    void create_camera(int camera_id);

public:
    Camera();
    ~Camera() = default;

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    int get_camera_id() const {
        return camera_id;
    }

    const CalibrationResult& get_calibration_result() const {
        return calib_res;
    }

    const cv::VideoCapture& get_camera() const {
        return cap;
    }

    cv::VideoCapture& get_camera() {
        return const_cast<cv::VideoCapture&>(
            static_cast<const Camera*>(this)->get_camera());
    }

    cv::Mat read_frame();
};

struct FileNotFoundException : public std::runtime_error {
public:
    FileNotFoundException(const std::string& file_path)
        : runtime_error("File not found: " + file_path) {}
};

struct CameraOpenException : public std::runtime_error {
    CameraOpenException(int camera_id)
        : std::runtime_error(fmt::format(
              "The camera with id {} cannot be opened.", camera_id)) {}
};

struct FrameCaptureException : public std::runtime_error {
    FrameCaptureException()
        : std::runtime_error("The camera cannot capture frames.") {}
};
