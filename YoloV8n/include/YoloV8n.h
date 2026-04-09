#include <trt_wrapper/common.h>
#include <trt_wrapper/Inference.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/Processor.h>

struct YoloDetectResult {
    int class_id;
    float confidence;
    cv::Rect box;
};

struct YoloDetectResultNMS {
    int32_t num_dets;
    float det_boxes[100][4];
    float det_scores[100];
    int32_t det_classes[100];
};

class YoloV8n {
private:
    std::string onnx_path;
    std::string engine_path;

    TRTPtr<nvinfer1::ICudaEngine> engine;
    TRTPtr<TRTInference> inference;

    // 模型是否启用了EfficientNMS插件
    bool enable_efficient_nms;

    float* input_buffer;
    float* output_buffer;

    size_t input_size;
    size_t output_size;

    void* gpu_input;
    void* gpu_output;

    std::vector<std::string> labels;
    void read_labels(const std::string& file_path);

    void set_tensor_addresses();

    std::vector<YoloDetectResult> decode_output();

    void apply_nms(std::vector<YoloDetectResult>& results,
                   float iou_threshold = 0.5f);

    void apply_deletterbox(std::vector<YoloDetectResult>& results);

    std::vector<YoloDetectResult> decode_output_nms();

    void preprocess(const cv::Mat& input);
    std::vector<YoloDetectResult> postprocess();
    void infer();
    auto InputData();
    auto OutputData();

public:
    YoloV8n(std::string onnx_path, std::string engine_path, TRTLogger& logger,
            bool enable_efficient_nms_plugin, bool always_rebuild = false);
    ~YoloV8n() noexcept;
    std::vector<YoloDetectResult> infer(const cv::Mat& input);
    cv::Mat visualize(const cv::Mat& input,
                      const std::vector<YoloDetectResult>& results);
};
