#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

#include <Cifar100CNN.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ImgItem {
    std::string file_name;
    int index;
    std::string label;

    ImgItem(const std::string& file_name, int index, const std::string& label)
        : file_name(file_name), index(index), label(label) {}
};

const int DATASET_SIZE = 10000;

int main() {
    std::string onnxModelPath =
        fs::path(PROJECT_SOURCE_DIR) / "model/cifar100_vgg.onnx";
    std::string engineFilePath =
        fs::path(PROJECT_SOURCE_DIR) / "model/cifar100_vgg.engine";
    TRTLogger glogger;

    Cifar100CNN cifar100(onnxModelPath, engineFilePath, glogger);

    std::ifstream metadata_file(fs::path(PROJECT_SOURCE_DIR) / "metadata.json");
    json metadata;
    metadata_file >> metadata;

    std::vector<ImgItem> items;
    items.reserve(DATASET_SIZE);
    std::vector<cv::Mat> images;
    images.reserve(DATASET_SIZE);
    for (const auto& item : metadata) {
        std::string img_path =
            fs::path(PROJECT_SOURCE_DIR) / "images" / item["file_name"];

        items.emplace_back(item["file_name"], item["class_index"],
                           item["class_name"]);

        images.emplace_back(cv::imread(img_path));
    }

    int correct_count = 0;
    std::vector<int> results = cifar100.infer(images, 256);

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] == items[i].index) {
            ++correct_count;
        }
    }

    std::cout << correct_count << std::endl;

    return 0;
}
