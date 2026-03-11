#include <filesystem>
#include "trt_wrapper/ModelBuilder.h"

#include <trt_wrapper/common.h>
#include <trt_wrapper/Inference.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/Processor.h>

namespace fs = std::filesystem;

class Cifar100CNN {
private:
    using Input = TensorSpec<32, 32, 3>;
    using Output = TensorSpec<100>;
    using Mean = ArraySpec<0.5071f, 0.4865f, 0.4409f>;
    using Std = ArraySpec<0.2673f, 0.2564f, 0.2761f>;
    using ChannelMap = ChannelMapSpec<2, 1, 0>;

    using Cifar100Processor = ConvertHWC2CHW<Input, Mean, Std, ChannelMap>;

    std::string onnx_path;
    std::string engine_path;

    TRTPtr<nvinfer1::ICudaEngine> engine;
    TRTPtr<TRTInference<Input, Output>> inference;

    float* input_buffer;
    float* output_buffer;

    void* gpu_input;
    void* gpu_output;

    void set_tensor_addresses() {
        inference->set_tensor_address("input", gpu_input);
        inference->set_tensor_address("output", gpu_output);
    }

    void preprocess(const cv::Mat& input) {
        Cifar100Processor::process(input, input_buffer);
    }

    int postprocess() {
        int class_id = std::distance(
            output_buffer,
            std::max_element(output_buffer,
                             output_buffer + Output::total_size()));
        return class_id;
    }

    auto InputData() {
        return cudaMemcpyAsync(gpu_input, input_buffer,
                               sizeof(float) * Input::total_size(),
                               cudaMemcpyHostToDevice, inference->get_stream());
    }

    auto OutputData() {
        return cudaMemcpyAsync(output_buffer, gpu_output,
                               sizeof(float) * Output::total_size(),
                               cudaMemcpyDeviceToHost, inference->get_stream());
    }

public:
    Cifar100CNN(std::string onnx_path, std::string engine_path,
                TRTLogger& logger)
        : onnx_path(onnx_path), engine_path(engine_path) {
        engine = TRTModelBuilder(logger).getEngine(onnx_path, engine_path);
        inference = TRTPtr<TRTInference<Input, Output>>(
            new TRTInference<Input, Output>(*engine));

        cudaHostAlloc((void**) &input_buffer,
                      sizeof(float) * Input::total_size(),
                      cudaHostAllocDefault);
        cudaHostAlloc((void**) &output_buffer,
                      sizeof(float) * Output::total_size(),
                      cudaHostAllocDefault);
        cudaMalloc(&gpu_input, sizeof(float) * Input::total_size());
        cudaMalloc(&gpu_output, sizeof(float) * Output::total_size());

        set_tensor_addresses();
    }

    ~Cifar100CNN() noexcept {
        cudaFree(gpu_input);
        cudaFree(gpu_output);
        cudaFreeHost(input_buffer);
        cudaFreeHost(output_buffer);
    }

    std::vector<int> infer(const std::vector<cv::Mat>& input) {
        std::vector<int> res;
        res.reserve(input.size());

        // 预处理第0张
        preprocess(input[0]);
        // 传入数据
        InputData();
        // 推理
        inference->infer();

        for (size_t i = 1; i < input.size(); ++i) {
            // 传回上一帧数据
            OutputData();
            // 预处理当前帧
            preprocess(input[i]);
            // 传入当前帧数据
            InputData();
            // 推理当前帧
            inference->infer();
            // 等待当前帧推理完成
            cudaStreamSynchronize(inference->get_stream());
            // 后处理上一帧数据
            res.push_back(postprocess());
        }

        // 传回最后一帧数据
        OutputData();
        // 等待传回完成
        cudaStreamSynchronize(inference->get_stream());
        // 最后一帧后处理
        res.push_back(postprocess());

        return res;
    }
};
