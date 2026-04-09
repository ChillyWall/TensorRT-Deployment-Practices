#include <trt_wrapper/common.h>
#include <trt_wrapper/Inference.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/Processor.h>

class Cifar100CNN {
private:
    std::string onnx_path;
    std::string engine_path;

    TRTPtr<nvinfer1::ICudaEngine> engine;
    TRTPtr<TRTInference> inference;

    float* input_buffer;
    float* output_buffer;

    void* gpu_input;
    void* gpu_output;

    void set_tensor_addresses();

    void preprocess(std::vector<cv::Mat>::const_iterator input,
                    size_t batch_size);
    void postprocess(std::vector<int>::iterator output, size_t batch_size);
    void infer(size_t batch_size);
    auto InputData(size_t batch_size);
    auto OutputData(size_t batch_size);

public:
    Cifar100CNN(std::string onnx_path, std::string engine_path,
                TRTLogger& logger, bool always_rebuild = false);
    ~Cifar100CNN() noexcept;
    std::vector<int> infer(const std::vector<cv::Mat>& input,
                           size_t batch_size = 0);
};
