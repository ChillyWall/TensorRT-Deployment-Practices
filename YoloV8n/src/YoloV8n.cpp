#include "trt_wrapper/ModelBuilder.h"

#include <memory>
#include "YoloV8n.h"

using InputImg = TensorSpec<640, 640, 3>;
using OutputRes = TensorSpec<84, 8400>;
using OutputDetClasses = TensorSpec<100>;
using OutputDetScores = TensorSpec<100>;
using OutputDetBoxes = TensorSpec<100, 4>;
using OutputNumDets = TensorSpec<1>;
using Mean = FloatArraySpec<0.f, 0.f, 0.f>;
using Std = FloatArraySpec<1.f, 1.f, 1.f>;
using ChannelMap = ChannelMapSpec<2, 1, 0>;

using YoloV8nProcessor =
    ConvertHWC2CHWAndNormalize<InputImg, Mean, Std, ChannelMap>;

using InputSize = TensorSpec<1280, 720>;
using TargetSize = TensorSpec<640, 640>;

void YoloV8n::set_tensor_addresses() {
    // 分配显存和内存，并绑定输入缓冲区地址
    input_size = sizeof(float) * InputImg::total_size();
    cudaHostAlloc((void**) &input_buffer, input_size, cudaHostAllocDefault);
    cudaMalloc(&gpu_input, input_size);
    inference->set_tensor_address("images", gpu_input);

    // 根据是否启用EfficientNMS插件，分配不同的输出缓冲区，并绑定输出地址
    if (enable_efficient_nms) {
        output_size = sizeof(YoloDetectResultNMS);
    } else {
        output_size = sizeof(float) * OutputRes::total_size();
    }

    cudaHostAlloc((void**) &output_buffer, output_size, cudaHostAllocDefault);
    cudaMalloc(&gpu_output, output_size);

    if (enable_efficient_nms) {
        inference->set_tensor_address(
            "num_dets",
            &(reinterpret_cast<YoloDetectResultNMS*>(gpu_output)->num_dets));
        inference->set_tensor_address(
            "det_boxes",
            &(reinterpret_cast<YoloDetectResultNMS*>(gpu_output)->det_boxes));
        inference->set_tensor_address(
            "det_scores",
            &(reinterpret_cast<YoloDetectResultNMS*>(gpu_output)->det_scores));
        inference->set_tensor_address(
            "det_classes",
            &(reinterpret_cast<YoloDetectResultNMS*>(gpu_output)->det_classes));
    } else {
        inference->set_tensor_address("output0", gpu_output);
    }
}

void YoloV8n::read_labels(const std::string& labels_path) {
    cv::FileStorage fs(labels_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error(
            std::format("File not found: {}", labels_path));
    }

    cv::FileNode node = fs["labels"];
    if (node.isSeq()) {
        labels.clear();
        labels.reserve(node.size());
        for (const auto& label : node) {
            labels.push_back((std::string) label);
        }
    } else {
        throw std::runtime_error(
            std::format("Invalid format in labels file: {}", labels_path));
    }
}

void YoloV8n::preprocess(const cv::Mat& input) {
    cv::Mat resized = LetterBox<InputSize, TargetSize>::process(input);
    YoloV8nProcessor::process(resized, input_buffer);
}

std::vector<YoloDetectResult> YoloV8n::decode_output() {
    std::vector<YoloDetectResult> results;
    // 预留50个结果的空间，避免频繁扩容
    results.reserve(50);
    constexpr int COLS = OutputRes::dims()[1];
    constexpr int ROWS = OutputRes::dims()[0];

    // 过滤低置信度的检测结果，并提取边界框和类别信息
    // 采用跳跃式访问，而非先转置输出矩阵，避免不必要的内存复制
    // 使用OpenMP进行并行处理，提升性能
#pragma omp parallel
    {
        // 每个线程创建自己的局部 vector
        std::vector<YoloDetectResult> local_results;
        local_results.reserve(10);  // 预留少量空间

#pragma omp for nowait  // nowait 减少同步开销
        for (size_t i = 0; i < COLS; ++i) {
            float max_score = 0;
            int class_id = -1;
            for (size_t j = 4; j < ROWS; ++j) {
                float s = output_buffer[j * COLS + i];
                if (s > max_score) {
                    max_score = s;
                    class_id = j - 4;
                }
            }
            if (max_score > 0.25f) {
                float x_center = output_buffer[0 * COLS + i];
                float y_center = output_buffer[1 * COLS + i];
                float width = output_buffer[2 * COLS + i];
                float height = output_buffer[3 * COLS + i];
                cv::Rect box(x_center - width / 2, y_center - height / 2, width,
                             height);
                local_results.push_back({class_id, max_score, box});
            }
        }

// 关键区域：将局部结果安全地合并到主结果
#pragma omp critical
        {
            results.insert(results.end(), local_results.begin(),
                           local_results.end());
        }
    }

    return results;
}

static inline float calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    float inter_area = (a & b).area();
    if (inter_area <= 0)
        return 0.f;
    float res =
        inter_area / static_cast<float>(a.area() + b.area() - inter_area);
    return res;
}

void YoloV8n::apply_nms(std::vector<YoloDetectResult>& candidates,
                        float iou_threshold) {
    if (candidates.empty())
        return;

    // 按置信率排序
    std::sort(candidates.begin(), candidates.end(),
              [](const YoloDetectResult& a, const YoloDetectResult& b) -> bool {
                  return a.confidence > b.confidence;
              });

    // 将要被删除的框的置信率设为负值，避免额外的 bool 标记空间
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].confidence < 0)
            continue;  // 已经被标记删除

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (candidates[j].confidence < 0)
                continue;

            if (candidates[i].class_id == candidates[j].class_id) {
                if (calculate_iou(candidates[i].box, candidates[j].box) >
                    iou_threshold) {
                    // 利用置信率字段作为标记位，负值表示被删除
                    candidates[j].confidence = -1.0f;
                }
            }
        }
    }

    // 删除被标记的框，即置信率被设为负值的项
    std::erase_if(candidates, [](const auto& d) { return d.confidence < 0; });
}

void YoloV8n::apply_deletterbox(std::vector<YoloDetectResult>& results) {
    for (auto& res : results) {
        res.box = DeLetterBox<InputSize, TargetSize>::process(res.box);
    }
}

std::vector<YoloDetectResult> YoloV8n::decode_output_nms() {
    auto output = reinterpret_cast<YoloDetectResultNMS*>(output_buffer);
    static_assert(sizeof(YoloDetectResultNMS) == (1 + 100 * 4 + 100 + 100) * 4);

    std::vector<YoloDetectResult> results;
    results.reserve(output->num_dets);
    for (int i = 0; i < output->num_dets; ++i) {
        int class_id = static_cast<int>(output->det_classes[i]);
        float confidence = output->det_scores[i];
        float x1 = output->det_boxes[i][0];
        float y1 = output->det_boxes[i][1];
        float x2 = output->det_boxes[i][2];
        float y2 = output->det_boxes[i][3];
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        results.push_back({class_id, confidence, box});
    }

    return results;
}

std::vector<YoloDetectResult> YoloV8n::postprocess() {
    std::vector<YoloDetectResult> results;
    if (enable_efficient_nms) {
        results = decode_output_nms();
    } else {
        results = decode_output();
        apply_nms(results, 0.45f);
    }
    apply_deletterbox(results);
    return results;
}

cv::Mat YoloV8n::visualize(const cv::Mat& input,
                           const std::vector<YoloDetectResult>& results) {
    cv::Mat image = input.clone();
    for (auto res : results) {
        float score = res.confidence;
        const auto& label = labels[res.class_id];
        std::string label_text = label + " " + cv::format("%.3f", score);

        // 绘制矩形和标签
        int base_line;
        cv::Size label_size = cv::getTextSize(
            label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, res.box.tl(), res.box.br(),
                      cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image,
                      cv::Point(res.box.x, res.box.y - label_size.height),
                      cv::Point(res.box.x + label_size.width, res.box.y),
                      cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, res.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(253, 168, 208), 1);
    }
    return image;
}

void YoloV8n::infer() {
    inference->infer();
}

auto YoloV8n::InputData() {
    return cudaMemcpyAsync(gpu_input, input_buffer, input_size,
                           cudaMemcpyHostToDevice, inference->get_stream());
}

auto YoloV8n::OutputData() {
    return cudaMemcpyAsync(output_buffer, gpu_output, output_size,
                           cudaMemcpyDeviceToHost, inference->get_stream());
}

YoloV8n::YoloV8n(std::string onnx_path, std::string engine_path,
                 TRTLogger& logger, bool enable_efficient_nms_plugin,
                 bool always_rebuild)
    : onnx_path(onnx_path),
      engine_path(engine_path),
      enable_efficient_nms(enable_efficient_nms_plugin) {
    auto builder = TRTModelBuilder(logger);
    if (always_rebuild || !(engine = builder.loadFromPlan(engine_path))) {
        engine = builder.buildFromOnnx(
            onnx_path, engine_path,
            [](nvinfer1::IBuilderConfig* config,
               nvinfer1::INetworkDefinition* network,
               nvinfer1::IBuilder* builder) {
                // 2. 精度设置：虽然 kFP16 弃用，但在 10.0 中作为 BuilderFlag
                // 依然是生效的（会有警告）
                if (builder->platformHasFastFp16()) {
                    config->setFlag(nvinfer1::BuilderFlag::kFP16);
                }
            });
    }
    inference = TRTPtr<TRTInference>(new TRTInference(*engine));

    set_tensor_addresses();

    read_labels(std::string(PACKAGE_ROOT_DIR) + "/config/labels.yml");
}

YoloV8n::~YoloV8n() noexcept {
    cudaFreeHost(input_buffer);
    cudaFreeHost(output_buffer);
    cudaFree(gpu_input);
    cudaFree(gpu_output);
}

std::vector<YoloDetectResult> YoloV8n::infer(const cv::Mat& input) {
    preprocess(input);
    InputData();
    infer();
    OutputData();
    cudaStreamSynchronize(inference->get_stream());
    return postprocess();
}
