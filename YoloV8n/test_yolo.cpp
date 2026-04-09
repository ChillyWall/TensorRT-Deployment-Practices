#include <filesystem>

#include <opencv2/opencv.hpp>

#include <camera.h>
#include <YoloV8n.h>

const int DATASET_SIZE = 10000;
namespace fs = std::filesystem;

int main() {
    std::string onnxModelPath =
        fs::path(PROJECT_SOURCE_DIR) / "model/yolov8n.onnx";
    std::string engineFilePath =
        fs::path(PROJECT_SOURCE_DIR) / "model/yolov8n.engine";
    TRTLogger glogger;

    YoloV8n yolo(onnxModelPath, engineFilePath, glogger);
    Camera camera;

    while (1) {
        cv::Mat frame = camera.read_frame();
        auto res = yolo.infer(frame);
        cv::Mat res_img = yolo.visualize(frame, res);
        cv::imshow("result", res_img);

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    return 0;
}
