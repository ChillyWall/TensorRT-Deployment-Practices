#include <fstream>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <trt_wrapper/common.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/ModelBuilder.h>

TRTModelBuilder::TRTModelBuilder(nvinfer1::ILogger& logger) : m_logger(logger) {
    initLibNvInferPlugins(&m_logger, "");
}

TRTPtr<nvinfer1::ICudaEngine>
TRTModelBuilder::loadFromPlan(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good())
        return nullptr;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    auto runtime =
        TRTPtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    return TRTPtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(data.data(), size));
}

TRTPtr<nvinfer1::ICudaEngine>
TRTModelBuilder::buildFromOnnx(const std::string& onnxPath,
                               const std::string& enginePath,
                               TRTBuildConfigFun configFun) {
    auto builder =
        TRTPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    auto network =
        TRTPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto config =
        TRTPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = TRTPtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));

    // 如果解析失败，说明 onnx 模型有问题
    if (!parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return nullptr;
    }

    // bool res = initLibNvInferPlugins(&m_logger, "");
    // if (!res) {
    //     m_logger.log(nvinfer1::ILogger::Severity::kERROR,
    //                  "Failed to initialize TensorRT plugins");
    // }

    if (configFun) {
        // 调用配置函数
        configFun(config.get(), network.get(), builder.get());
    }

    // 编译模型
    auto plan = TRTPtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    // 将编译好的 Engine 保存到磁盘，下次直接 load
    std::ofstream outfile(enginePath, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    auto runtime =
        TRTPtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    return TRTPtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}
