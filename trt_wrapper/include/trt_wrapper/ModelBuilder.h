#pragma once

#include <string>

#include <NvInfer.h>

#include "trt_wrapper/common.h"

class TRTModelBuilder {
public:
    TRTModelBuilder(nvinfer1::ILogger& logger) : m_logger(logger) {}

    /**
     * @brief 获取engine
     *
     * @param onnxPath onnx模型文件的地址
     * @param enginePath engine文件的地址
     * @return 指向engine的unique_ptr指针
     */
    TRTPtr<nvinfer1::ICudaEngine> getEngine(const std::string& onnxPath,
                                            const std::string& enginePath);

private:
    // 从本地 .engine (Plan) 文件加载
    TRTPtr<nvinfer1::ICudaEngine> loadFromPlan(const std::string& enginePath);

    // 从 ONNX 编译并保存为 .engine
    TRTPtr<nvinfer1::ICudaEngine> buildFromOnnx(const std::string& onnxPath,
                                                const std::string& enginePath);
    nvinfer1::ILogger& m_logger;
};
