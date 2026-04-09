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

template <TensorSpecType InputSpec, FloatArraySpecType Mean,
          FloatArraySpecType Std, ChannelMapType ChannelMap = KeepChannelMap>
class ConvertHWC2CHWAndNormalize {
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

template <TensorSpecType InputSpec, TensorSpecType OutputSpec>
struct LetterBox {
    static cv::Mat process(const cv::Mat& input) {
        constexpr int input_width = InputSpec::dims()[0];
        constexpr int input_height = InputSpec::dims()[1];
        constexpr int output_width = OutputSpec::dims()[0];
        constexpr int output_height = OutputSpec::dims()[1];

        constexpr float scale =
            std::min(static_cast<float>(output_width) / input_width,
                     static_cast<float>(output_height) / input_height);

        constexpr int resized_width = static_cast<int>(input_width * scale);
        constexpr int resized_height = static_cast<int>(input_height * scale);

        constexpr int x_offset = (output_width - resized_width) / 2;
        constexpr int y_offset = (output_height - resized_height) / 2;

        cv::Mat resized;
        cv::resize(input, resized, cv::Size(resized_width, resized_height));

        cv::Mat output =
            cv::Mat::zeros(output_height, output_width, input.type());

        cv::Rect roi(x_offset, y_offset, resized_width, resized_height);
        resized.copyTo(output(roi));

        return output;
    }
};

template <TensorSpecType InputSpec, TensorSpecType OutputSpec>
struct DeLetterBox {
    static cv::Rect process(const cv::Rect& input) {
        constexpr int input_width = InputSpec::dims()[0];
        constexpr int input_height = InputSpec::dims()[1];
        constexpr int output_width = OutputSpec::dims()[0];
        constexpr int output_height = OutputSpec::dims()[1];

        constexpr float scale =
            std::min(static_cast<float>(output_width) / input_width,
                     static_cast<float>(output_height) / input_height);

        constexpr int resized_width = static_cast<int>(input_width * scale);
        constexpr int resized_height = static_cast<int>(input_height * scale);

        constexpr int x_offset = (output_width - resized_width) / 2;
        constexpr int y_offset = (output_height - resized_height) / 2;

        return cv::Rect((input.x - x_offset) / scale,
                        (input.y - y_offset) / scale, input.width / scale,
                        input.height / scale);
    }
};
