#pragma once

#include <concepts>
#include <memory>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
#if NV_TENSORRT_MAJOR < 9
            obj->destroy();
#else
            delete obj;  // TensorRT 10.0+ 推荐做法
#endif
        }
    }
};

template <typename T>
using TRTPtr = std::unique_ptr<T, TRTDeleter>;

template <typename T>
concept Processor = requires(const cv::Mat& img, float* output) {
    { T::process(img, output) } -> std::same_as<void>;
};

// 张量规格，编译期维度信息
template <size_t... Sizes>
struct TensorSpec {
    static constexpr size_t total_size() {
        size_t size = 1;
        ((size *= Sizes), ...);
        return size;
    }
    static constexpr std::array<size_t, sizeof...(Sizes)> dims() {
        return {Sizes...};
    }
};

template <typename T>
concept TensorSpecType = requires {
    { T::total_size() } -> std::convertible_to<size_t>;
};

// 编译期数组
template <float... elems>
struct FloatArraySpec {
    static constexpr std::array<float, sizeof...(elems)> values() {
        return {elems...};
    }
};

/* 是是否编译期数组规格类型，即可通过values()方法获取std::array<float,
 * N>类型的数组，其中N为元素个数个数 */
template <typename T>
concept FloatArraySpecType = requires {
    {
        T::values()
    } -> std::convertible_to<std::array<float, T::values().size()>>;
};

// 颜色空间映射，编译器索引信息
// 注意：使用RGB表示第一个，第二个和第三个通道的索引位置，哪怕你不是要转成RGB
template <typename T>
concept ChannelMapType =
    requires {
        { T::r } -> std::convertible_to<int>;
        { T::g } -> std::convertible_to<int>;
        { T::b } -> std::convertible_to<int>;
        { T::index() } -> std::convertible_to<std::array<int, 3>>;
    } && (T::r >= 0 && T::r < 3) && (T::g >= 0 && T::g < 3) &&
    (T::b >= 0 && T::b < 3);

template <int R, int G, int B>
struct ChannelMapSpec {
    static constexpr int r = R;
    static constexpr int g = G;
    static constexpr int b = B;

    static constexpr std::array<int, 3> index() {
        return {R, G, B};
    }
};

using KeepChannelMap = ChannelMapSpec<0, 1, 2>;
