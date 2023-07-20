#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#include "tensorrt/common.h"
#include "tensorrt/converter.h"
#include "trt_bridge.h"

TrtBridge* TrtBridge::Create(const std::string& model_pwd, NetworkMeta* p_meta) {
    TrtBridge* p = new TrtBridge();
    std::string trt_model_pwd = std::string(model_pwd);
    if (model_pwd.find(".onnx") != std::string::npos) {
        trt_model_pwd = trt_model_pwd.replace(trt_model_pwd.find(".onnx"), std::string(".onnx").length(), ".trt\0");
        TrtConverter converter;
        converter.Onnx2Trt(model_pwd, trt_model_pwd, true);
    }
    p->model_pwd_ = trt_model_pwd;          // Why I could reset private variables here?
    p->network_meta_.reset(p_meta);
    return p;
}

int32_t TrtBridge::Initialize() {
    // 1. Deseralize trt model and create corresponding engine and context
    sample::Logger glogger;
    runtime_.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    std::ifstream stream(model_pwd_, std::ios::binary);
    std::string buffer;
    if (stream) {
        stream >> std::noskipws;
        std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));   // copy file contents from ifstream to string using iterator
    }
    engine_.reset((runtime_->deserializeCudaEngine(buffer.data(), buffer.size())));
    context_.reset((engine_->createExecutionContext()));

    // 2.1 Readin model's input and output metadata
    network_meta_->batch_size = engine_->getMaxBatchSize();
    int32_t input_meta_list_index = 0;
    int32_t output_meta_list_index = 0;
    std::map<int32_t, int32_t> input_indexs_gpu2meta;
    std::map<int32_t, int32_t> output_indexs_gpu2meta;
    for (auto& input_tensor_meta : network_meta_->input_tensor_meta_list) {
        const int gpu_input_index = engine_->getBindingIndex(input_tensor_meta.tensor_name.c_str());
        input_indexs_gpu2meta.insert({gpu_input_index, input_meta_list_index++});
        nvinfer1::Dims input_shape = engine_->getBindingDimensions(gpu_input_index);
        if (network_meta_->input_nchw) {
            input_tensor_meta.net_in_c = input_shape.d[1];
            input_tensor_meta.net_in_h = input_shape.d[2];
            input_tensor_meta.net_in_w = input_shape.d[3];
        } else {
            input_tensor_meta.net_in_h = input_shape.d[1];
            input_tensor_meta.net_in_w = input_shape.d[2];
            input_tensor_meta.net_in_c = input_shape.d[3];
        }
        input_tensor_meta.input_scale = 1.0;
        input_tensor_meta.net_in_elements = network_meta_->batch_size * input_tensor_meta.net_in_c * input_tensor_meta.net_in_h * input_tensor_meta.net_in_w;
    }
    for (auto& output_tensor_meta : network_meta_->output_tensor_meta_list) {
        const int gpu_output_index = engine_->getBindingIndex(output_tensor_meta.tensor_name.c_str());
        output_indexs_gpu2meta.insert({gpu_output_index, output_meta_list_index++});
        nvinfer1::Dims output_shape = engine_->getBindingDimensions(gpu_output_index);
        if (output_shape.nbDims == 3) {
            if (output_tensor_meta.output_nlc) {
                output_tensor_meta.net_out_l = output_shape.d[1];
                output_tensor_meta.net_out_c = output_shape.d[2];
            } else {
                output_tensor_meta.net_out_c = output_shape.d[1];
                output_tensor_meta.net_out_l = output_shape.d[2];
            }
        }
        if (output_shape.nbDims == 4) {
            if (output_tensor_meta.output_nlc) {
                output_tensor_meta.net_out_h = output_shape.d[1];
                output_tensor_meta.net_out_w = output_shape.d[2];
                output_tensor_meta.net_out_c = output_shape.d[3];
                output_tensor_meta.net_out_l = output_shape.d[1] * output_shape.d[2];
            } else {
                output_tensor_meta.net_out_c = output_shape.d[1];
                output_tensor_meta.net_out_h = output_shape.d[2];
                output_tensor_meta.net_out_w = output_shape.d[3];
                output_tensor_meta.net_out_l = output_shape.d[2] * output_shape.d[3];
            }
        }
        output_tensor_meta.net_out_elements = network_meta_->batch_size * output_tensor_meta.net_out_c * output_tensor_meta.net_out_l;
    }
    network_meta_->input_tensor_num = network_meta_->input_tensor_meta_list.size();
    network_meta_->output_tensor_num = network_meta_->output_tensor_meta_list.size();
    
    // 2.2 Rearrange input meta and output meta to match model's input and output order 
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        int32_t input_index = input_indexs_gpu2meta.find(i)->second;
        network_meta_->input_tensor_meta_list.push_back(network_meta_->input_tensor_meta_list[input_index]);
        network_meta_->input_name2index.insert({network_meta_->input_tensor_meta_list[input_index].tensor_name, i});
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        int32_t output_index = output_indexs_gpu2meta.find(i + network_meta_->input_tensor_num)->second;
        network_meta_->output_tensor_meta_list.push_back(network_meta_->output_tensor_meta_list[output_index]);
        network_meta_->output_name2index.insert({network_meta_->output_tensor_meta_list[output_index].tensor_name, i});
    }
    network_meta_->input_tensor_meta_list.assign(network_meta_->input_tensor_meta_list.begin()+network_meta_->input_tensor_num, network_meta_->input_tensor_meta_list.end());
    network_meta_->output_tensor_meta_list.assign(network_meta_->output_tensor_meta_list.begin()+network_meta_->output_tensor_num, network_meta_->output_tensor_meta_list.end());
    // TO DO CHECK

    // 3. Construct network's input and output space on systeam ram; Construct memory on gpu and bind them
    gpu_binding_input_ptrs_.resize(network_meta_->input_tensor_num);
    gpu_binding_output_ptrs_.resize(network_meta_->output_tensor_num);
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        const auto& tensor_meta = network_meta_->input_tensor_meta_list[i];
        input_ptrs_.push_back(new float[tensor_meta.net_in_elements]);
        cudaMalloc(&gpu_binding_input_ptrs_[i], tensor_meta.net_in_elements * sizeof(float));
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        const auto& tensor_meta = network_meta_->output_tensor_meta_list[i];
        output_ptrs_.push_back(new float[tensor_meta.net_out_elements]);
        cudaMalloc(&gpu_binding_output_ptrs_[i], tensor_meta.net_out_elements * sizeof(float));
    }
    bindings_.insert(bindings_.begin(), gpu_binding_input_ptrs_.begin(), gpu_binding_input_ptrs_.end());
    bindings_.insert(bindings_.end(), gpu_binding_output_ptrs_.begin(), gpu_binding_output_ptrs_.end());
    ConvertNormalizedParameters(network_meta_->normalize.mean, network_meta_->normalize.norm);

    return 1;
}

int32_t TrtBridge::ConvertNormalizedParameters(float* mean, float* norm) {
    // float scale = network_meta_->input_scale;
    float scale = 1.0;
    // Convert to speede up normalization:  
    // (((src/255) - mean)/norm)*scale ----> ((src - mean*255) / (255*norm))*scale ----> (src - mean*255) * (scale/(255*norm))
    for (int32_t i = 0; i < 3; i++) {
        mean[i] *= 255;
        norm[i] *= 255;
        norm[i] = scale / norm[i];
    }
    return 1;
}

template <typename T>
int32_t TrtBridge::PermuateAndNormalize(T* input_ptr, uint8_t* src, int32_t input_h, int32_t input_w, int32_t input_c) {
    // Convert NHWC to NCHW && Do normalized operation to the original input image.
    float* mean = network_meta_->normalize.mean;
    float* norm = network_meta_->normalize.norm;
    int32_t spatial_size = input_h * input_w;
    memset(input_ptr, 0, sizeof(T) * input_h * input_w * input_c);
    if (network_meta_->input_nchw) {
#pragma omp parallel for num_threads(4)
        for (int32_t c = 0; c < input_c; c++) {
            for (int32_t i = 0; i < spatial_size; i++) {
                input_ptr[spatial_size * c + i] = (src[i * input_c + c] - mean[c]) * norm[c];
                //  input_ptr[spatial_size * c + i] = src[i * input_c + c] - 128;
            }
        }
    } else {
#pragma omp parallel for num_threads(4)
        for (int32_t c = 0; c < input_c; c++) {
            for (int32_t i = 0; i < spatial_size; i++) {
                input_ptr[i * input_c + c] = (src[i * input_c + c] - mean[c]) * norm[c];
            }
        }
    }
    return 1;
}

int32_t TrtBridge::SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h, const kCropStyle style) {
    if (style == kCropStyle::CropAll_Coverup) {    // retain
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_1) {    // a very distinct one, shed 0.4 top part of src, disgard of ratio
        src_crop_.x = 0;
        src_crop_.y = src_h * 0.4;
        src_crop_.width = src_w;
        src_crop_.height = src_h - src_crop_.y;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_0) {    // shed top part of src, retain width and make the crop ratio equals to model's input's
        float src_ratio = 1.0 * src_h / src_w;
        float dst_ratio = 1.0 * dst_h / dst_w;
        if (src_ratio > dst_ratio) {
            src_crop_.width = src_w;
            src_crop_.height = static_cast<int32_t>(src_w * dst_ratio);
            src_crop_.x = 0;
            src_crop_.y = src_h - src_crop_.height;
        } else {
            src_crop_.width = src_w;
            src_crop_.height = src_h;
            src_crop_.x = 0;
            src_crop_.y = 0;
        }
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropAll_Embedd) {    // embedd src into dst's center, src's ratio not changed
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        float src_ratio = 1.0 * src_w / src_h;
        float dst_ratio = 1.0 * dst_w / dst_h;
        if (src_ratio > dst_ratio) {
            // Use dst's width as base
            dst_crop_.width = dst_w;
            // dst_crop_.height = dst_h * dst_ratio / src_ratio;
            // dst_crop_.height = src_h * (dst_w / src_w);
            dst_crop_.height = static_cast<int32_t>(dst_w / src_ratio);
            dst_crop_.x = 0;
            dst_crop_.y = (dst_h - dst_crop_.height) / 2;
        } else {
            // Use dst's height as base
            dst_crop_.height = dst_h;
            dst_crop_.width = static_cast<int32_t>(dst_h / src_ratio);
            dst_crop_.x = (dst_w - dst_crop_.width) / 2;
            dst_crop_.y = 0;
        }
    }
    return 1;
}

int32_t TrtBridge::PreProcess(cv::Mat& original_img, const kCropStyle& style) {
    auto tensor_meta = network_meta_->input_tensor_meta_list[0];
    int32_t input_h = tensor_meta.net_in_h;
    int32_t input_w = tensor_meta.net_in_w;
    int32_t input_c = tensor_meta.net_in_c;
    SetCropAttr(original_img.cols, original_img.rows, input_w, input_h, style);
    const auto& t0 = std::chrono::steady_clock::now();
    cv::Mat sample = cv::Mat::zeros(input_h, input_w, CV_8UC3); // (h, w)
    cv::Mat resized_mat = sample(dst_crop_);
    cv::resize(original_img(src_crop_), resized_mat, resized_mat.size(), 0, 0, cv::INTER_NEAREST); // Why must assign fx and fy to enable deep copy?
    // cv::imwrite("./resized2.jpg", sample);
    if (network_meta_->input_rgb) {
        cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
    }
    uint8_t* src = (uint8_t*)sample.data;
    const auto& t1 = std::chrono::steady_clock::now();
    PermuateAndNormalize((float*)input_ptrs_[0], src, input_h, input_w, input_c);
    const auto& t2 = std::chrono::steady_clock::now();
    // std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
    // std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
    return 1;
}

int32_t TrtBridge::PreProcess(cv::Mat& original_img, const std::string& input_name, const kCropStyle& style) {
    int32_t input_index = network_meta_->input_name2index.find(input_name)->second;
    auto tensor_meta = network_meta_->input_tensor_meta_list[input_index];
    int32_t input_h = tensor_meta.net_in_h;
    int32_t input_w = tensor_meta.net_in_w;
    int32_t input_c = tensor_meta.net_in_c;
    SetCropAttr(original_img.cols, original_img.rows, input_w, input_h, style);
    const auto& t0 = std::chrono::steady_clock::now();
    cv::Mat sample = cv::Mat::zeros(input_w, input_h, CV_8UC3);
    cv::Mat resized_mat = sample(dst_crop_);
    cv::resize(original_img, resized_mat, resized_mat.size(), 0, 0, cv::INTER_NEAREST); // Why must assign fx and fy to enable deep copy?
    if (network_meta_->input_rgb) {
        cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
    }
    uint8_t* src = (uint8_t*)sample.data;
    const auto& t1 = std::chrono::steady_clock::now();
    PermuateAndNormalize((float*)input_ptrs_[input_index], src, input_h, input_w, input_c);
    const auto& t2 = std::chrono::steady_clock::now();
    // std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
    // std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
    return 1;
}

int32_t TrtBridge::Inference() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        cudaMemcpyAsync(gpu_binding_input_ptrs_[i], input_ptrs_[i], network_meta_->input_tensor_meta_list[i].net_in_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    context_->enqueue(network_meta_->batch_size, &bindings_[0], stream, NULL);
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        cudaMemcpyAsync(output_ptrs_[i], gpu_binding_output_ptrs_[i], network_meta_->output_tensor_meta_list[i].net_out_elements * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 1;
}

int32_t TrtBridge::Finalize() {
    for (auto& ptr : input_ptrs_) {
        delete[] (float*)ptr;
    }
    for (auto& ptr : output_ptrs_) {
        delete[] (float*)ptr;
    }

    for (auto& bind_ptr : bindings_) {
        cudaFree(bind_ptr);
    }

    return 1;
}