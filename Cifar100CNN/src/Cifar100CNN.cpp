#include <cstdint>
#include <format>

#include "trt_wrapper/ModelBuilder.h"

#include "Cifar100CNN.h"

using InputImg = TensorSpec<32, 32, 3>;
using OutputRes = TensorSpec<100>;
using Mean = FloatArraySpec<0.5071f, 0.4865f, 0.4409f>;
using Std = FloatArraySpec<0.2673f, 0.2564f, 0.2761f>;
using ChannelMap = ChannelMapSpec<2, 1, 0>;
using Cifar100Processor =
    ConvertHWC2CHWAndNormalize<InputImg, Mean, Std, ChannelMap>;

using BatchSize = TensorSpec<1, 64, 256>;
using Input = TensorSpec<BatchSize::dims()[2], 3, 32, 32>;
using Output = TensorSpec<BatchSize::dims()[2], 100>;

void Cifar100CNN::set_tensor_addresses() {
    inference->set_tensor_address("input", gpu_input);
    inference->set_tensor_address("output", gpu_output);
}

void Cifar100CNN::preprocess(std::vector<cv::Mat>::const_iterator input,
                             size_t batch_size) {
    size_t img_size = InputImg::total_size();

#pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        Cifar100Processor::process(*(input + i), input_buffer + i * img_size);
    }
}

void Cifar100CNN::postprocess(std::vector<int>::iterator output,
                              size_t batch_size) {
    size_t res_size = OutputRes::total_size();

#pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        float* output_buffer_idx = output_buffer + i * res_size;
        int class_id = std::distance(
            output_buffer_idx,
            std::max_element(output_buffer_idx, output_buffer_idx + res_size));
        *(output + i) = class_id;
    }
}

void Cifar100CNN::infer(size_t batch_size) {
    inference->infer([batch_size](nvinfer1::IExecutionContext* context) {
        context->setInputShape(
            "input", nvinfer1::Dims4 {(int64_t) batch_size, 3, 32, 32});
    });
}

auto Cifar100CNN::InputData(size_t batch_size) {
    return cudaMemcpyAsync(gpu_input, input_buffer,
                           sizeof(float) * InputImg::total_size() * batch_size,
                           cudaMemcpyHostToDevice, inference->get_stream());
}

auto Cifar100CNN::OutputData(size_t batch_size) {
    return cudaMemcpyAsync(output_buffer, gpu_output,
                           sizeof(float) * OutputRes::total_size() * batch_size,
                           cudaMemcpyDeviceToHost, inference->get_stream());
}

Cifar100CNN::Cifar100CNN(std::string onnx_path, std::string engine_path,
                         TRTLogger& logger, bool always_rebuild)
    : onnx_path(onnx_path), engine_path(engine_path) {
    auto builder = TRTModelBuilder(logger);
    if (always_rebuild || !(engine = builder.loadFromPlan(engine_path))) {
        engine = builder.buildFromOnnx(
            onnx_path, engine_path,
            [](nvinfer1::IBuilderConfig* config,
               nvinfer1::INetworkDefinition* network,
               nvinfer1::IBuilder* builder) {
                auto profile = builder->createOptimizationProfile();
                const char* inputName = network->getInput(0)->getName();
                auto batch_sizes = BatchSize::dims();
                // [Min, Opt, Max]
                profile->setDimensions(
                    inputName, nvinfer1::OptProfileSelector::kMIN,
                    nvinfer1::Dims4 {(int64_t) batch_sizes[0], 3, 32, 32});
                profile->setDimensions(
                    inputName, nvinfer1::OptProfileSelector::kOPT,
                    nvinfer1::Dims4 {(int64_t) batch_sizes[1], 3, 32, 32});
                profile->setDimensions(
                    inputName, nvinfer1::OptProfileSelector::kMAX,
                    nvinfer1::Dims4 {(int64_t) batch_sizes[2], 3, 32, 32});
                config->addOptimizationProfile(profile);

                // 2. 精度设置：虽然 kFP16 弃用，但在 10.0 中作为 BuilderFlag
                // 依然是生效的（会有警告）
                if (builder->platformHasFastFp16()) {
                    config->setFlag(nvinfer1::BuilderFlag::kFP16);
                }
            });
    }
    inference = TRTPtr<TRTInference>(new TRTInference(*engine));

    cudaHostAlloc((void**) &input_buffer, sizeof(float) * Input::total_size(),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**) &output_buffer, sizeof(float) * Output::total_size(),
                  cudaHostAllocDefault);
    cudaMalloc(&gpu_input, sizeof(float) * Input::total_size());
    cudaMalloc(&gpu_output, sizeof(float) * Output::total_size());

    set_tensor_addresses();
}

Cifar100CNN::~Cifar100CNN() noexcept {
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFreeHost(input_buffer);
    cudaFreeHost(output_buffer);
}

std::vector<int> Cifar100CNN::infer(const std::vector<cv::Mat>& input,
                                    size_t batch_size) {
    size_t input_size = input.size();
    std::vector<int> res(input_size);

    auto batch_sizes = BatchSize::dims();
    if (batch_size == 0) {
        // 使用默认的最优 batch size
        batch_size = batch_sizes[1];
    } else if (batch_size > batch_sizes[2]) {
        // 若超过则使用最大 batch size
        batch_size = batch_sizes[2];
    }

    size_t batches =
        input_size / batch_size + (((input_size % batch_size) == 0) ? 0 : 1);

    for (size_t i = 0; i < batches; ++i) {
        size_t cur_batch_size =
            std::min(batch_size, input_size - i * batch_size);
        preprocess(input.cbegin() + i * batch_size, cur_batch_size);
        InputData(cur_batch_size);
        infer(cur_batch_size);
        OutputData(cur_batch_size);
        cudaStreamSynchronize(inference->get_stream());
        postprocess(res.begin() + i * batch_size, cur_batch_size);

        std::cout << std::format("batch {} with size {} finished\n", i,
                                 cur_batch_size);
    }

    return res;
}
