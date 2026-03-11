#include <fstream>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <trt_wrapper/common.h>
#include <trt_wrapper/logger.h>
#include <trt_wrapper/ModelBuilder.h>

TRTPtr<nvinfer1::ICudaEngine>
TRTModelBuilder::getEngine(const std::string& onnxPath,
                           const std::string& enginePath) {
    // 1. 尝试从本地加载
    auto engine = loadFromPlan(enginePath);
    if (engine) {
        m_logger.log(
            TRTLogger::Severity::kINFO,
            fmt::format("[TRT] Loaded engine from: {}", enginePath).c_str());
        return engine;
    }

    // 2. 加载失败则编译
    m_logger.log(
        TRTLogger::Severity::kINFO,
        fmt::format("[TRT] Engine not found. Building from ONNX: {}", onnxPath)
            .c_str());
    return buildFromOnnx(onnxPath, enginePath);
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
                               const std::string& enginePath) {
    auto builder =
        TRTPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTPtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    auto config =
        TRTPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = TRTPtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));

    if (!parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return nullptr;
    }

    // 这里可以开启 FP16 优化
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
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
