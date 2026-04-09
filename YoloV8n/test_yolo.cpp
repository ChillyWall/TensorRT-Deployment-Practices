#include <filesystem>

#include <opencv2/opencv.hpp>

#include <camera.h>
#include <YoloV8n.h>

const int DATASET_SIZE = 10000;
namespace fs = std::filesystem;

int main() {
    cv::FileStorage fs(fs::path(PACKAGE_ROOT_DIR) / "config/config.yml",
                       cv::FileStorage::READ);
    std::string onnxModelPath =
        fs::path(PROJECT_SOURCE_DIR) / std::string(fs["onnx_path"]);
    std::string engineFilePath =
        fs::path(PROJECT_SOURCE_DIR) / std::string(fs["engine_path"]);
    TRTLogger glogger;

    int enable_efficient_nms_plugin = fs["enable_efficient_nms_plugin"];
    int always_rebuild = fs["always_rebuild"];

    std::cout << std::format(
                     "enable_efficient_nms_plugin: {}\nalways_rebuild: {}",
                     enable_efficient_nms_plugin, always_rebuild)
              << std::endl;

    YoloV8n yolo(onnxModelPath, engineFilePath, glogger,
                 enable_efficient_nms_plugin, always_rebuild);
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
