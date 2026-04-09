#include <trt_wrapper/common.h>
#include <trt_wrapper/Inference.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/Processor.h>

struct YoloDetectResult {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloV8n {
private:
    std::string onnx_path;
    std::string engine_path;

    TRTPtr<nvinfer1::ICudaEngine> engine;
    TRTPtr<TRTInference> inference;

    float* input_buffer;
    float* output_buffer;

    void* gpu_input;
    void* gpu_output;

    std::vector<std::string> labels;
    void read_labels(const std::string& file_path);

    void set_tensor_addresses();

    std::vector<YoloDetectResult> decode_output();
    void apply_nms(std::vector<YoloDetectResult>& results,
                   float iou_threshold = 0.5f);
    void apply_deletterbox(std::vector<YoloDetectResult>& results);

    void preprocess(const cv::Mat& input);
    std::vector<YoloDetectResult> postprocess();
    void infer();
    auto InputData();
    auto OutputData();

public:
    YoloV8n(std::string onnx_path, std::string engine_path, TRTLogger& logger,
            bool always_rebuild = false);
    ~YoloV8n() noexcept;
    std::vector<YoloDetectResult> infer(const cv::Mat& input);
    cv::Mat visualize(const cv::Mat& input,
                      const std::vector<YoloDetectResult>& results);
};
