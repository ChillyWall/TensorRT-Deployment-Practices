#pragma once

#include <NvInfer.h>

#include <trt_wrapper/common.h>

using TRTInferConfigFun = std::function<void(nvinfer1::IExecutionContext*)>;

class TRTInference {
protected:
    nvinfer1::ICudaEngine& engine;
    TRTPtr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;

public:
    TRTInference() = delete;

    TRTInference(nvinfer1::ICudaEngine& engine)
        : engine(engine), context(engine.createExecutionContext()) {
        cudaStreamCreate(&stream);
    }

    TRTInference(const TRTInference&) = delete;
    TRTInference& operator=(const TRTInference&) = delete;
    TRTInference(TRTInference&&) noexcept = delete;
    TRTInference& operator=(TRTInference&&) noexcept = delete;

    ~TRTInference() {
        cudaStreamDestroy(stream);
    }

    template <typename... Args>
    bool set_tensor_address(Args&&... args) {
        return context->setTensorAddress(std::forward<Args>(args)...);
    }

    template <typename... Args>
    const void* get_tensor_address(Args&&... args) {
        return context->getTensorAddress(std::forward<Args>(args)...);
    }

    cudaStream_t get_stream() {
        return stream;
    }

    bool infer(TRTInferConfigFun configFun = nullptr) {
        if (configFun) {
            configFun(context.get());
        }
        return context->enqueueV3(stream);
    }
};
