#pragma once

#include <fmt/format.h>
#include <NvInfer.h>
#include <iostream>

// --- [日志记录器] ---
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << fmt::format("[TRT] {}", msg) << std::endl;
    }
};
