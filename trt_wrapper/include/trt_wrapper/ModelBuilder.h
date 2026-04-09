#pragma once

#include <string>

#include <NvInfer.h>

#include "trt_wrapper/common.h"

using TRTBuildConfigFun =
    std::function<void(nvinfer1::IBuilderConfig*, nvinfer1::INetworkDefinition*,
                       nvinfer1::IBuilder*)>;

class TRTModelBuilder {
public:
    TRTModelBuilder(nvinfer1::ILogger& logger);
    // 从本地 .engine (Plan) 文件加载
    TRTPtr<nvinfer1::ICudaEngine> loadFromPlan(const std::string& enginePath);

    // 从 ONNX 编译并保存为 .engine
    TRTPtr<nvinfer1::ICudaEngine>
    buildFromOnnx(const std::string& onnxPath, const std::string& enginePath,
                  TRTBuildConfigFun configFun = nullptr);

private:
    nvinfer1::ILogger& m_logger;
};
