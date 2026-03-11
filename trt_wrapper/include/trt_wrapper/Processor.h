#include <array>

#include <opencv2/opencv.hpp>

#include <trt_wrapper/common.h>

template <size_t... Is>
constexpr auto make_alphas_impl(const std::array<float, sizeof...(Is)>& stds,
                                std::index_sequence<Is...>) {
    return std::array<float, sizeof...(Is)> {(1.0f / (255.0f * stds[Is]))...};
}

template <size_t... Is>
constexpr auto make_betas_impl(const std::array<float, sizeof...(Is)>& means,
                               const std::array<float, sizeof...(Is)>& stds,
                               std::index_sequence<Is...>) {
    return std::array<float, sizeof...(Is)> {(-means[Is] / stds[Is])...};
}

template <TensorSpecType InputSpec, ArraySpecType Mean, ArraySpecType Std,
          ChannelMapType ChannelMap = KeepChannelMap>
class ConvertHWC2CHW {
private:
    constexpr static int input_height = InputSpec::dims()[0];
    constexpr static int input_width = InputSpec::dims()[1];
    constexpr static int channel_num = InputSpec::dims()[2];
    constexpr static std::array<int, channel_num> channel_map =
        ChannelMap::index();

    constexpr static std::array<float, channel_num> alphas = make_alphas_impl(
        Std::values(), std::make_index_sequence<channel_num> {});
    constexpr static std::array<float, channel_num> betas =
        make_betas_impl(Mean::values(), Std::values(),
                        std::make_index_sequence<channel_num> {});

public:
    static void process(const cv::Mat& input, float* output) {
        int channel_size = input_height * input_width;

        std::array<cv::Mat, channel_num> bgr_channels;
        cv::split(input, bgr_channels);

        for (int i = 0; i < channel_num; ++i) {
            cv::Mat target_slice(
                input_height, input_width, CV_32FC1,
                output + ChannelMap::index()[i] * channel_size);

            bgr_channels[i].convertTo(target_slice, CV_32FC1,
                                      alphas[channel_map[i]],
                                      betas[channel_map[i]]);
        }
    }
};
