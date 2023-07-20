#include <iostream>
#include <memory>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "common.h"

#define OPT_MAX_WORK_SPACE_SIZE ((size_t)1 << 30)

class TrtConverter {
public:
    static int32_t Onnx2Trt (const std::string onnx_model_pwd, const std::string trt_model_pwd, bool use_fp16) {
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        config->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);
        if (use_fp16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser->parseFromFile(onnx_model_pwd.c_str(), (int32_t)nvinfer1::ILogger::Severity::kWARNING)) {
            std::cout << onnx_model_pwd << " parsing to trt failed" << std::endl;
            return 0;
        }
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            std::cout << onnx_model_pwd << " creating plan failed" << std::endl;
            return 0;
        }
        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) {
            std::cout << onnx_model_pwd << " creating engine failed" << std::endl;
            return 0;
        }

        // std::string trt_model_pwd = std::string(onnx_model_pwd);
        // trt_model_pwd = trt_model_pwd.replace(trt_model_pwd.find(".onnx"), std::string(".onnx").length(), ".trt\0");
        std::ofstream ofs(trt_model_pwd, std::ios::out | std::ios::binary);
        ofs.write((char*)(plan->data()), plan->size());
        ofs.close();

        return 1;
    }
};