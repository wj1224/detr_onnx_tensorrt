#include <NvOnnxParser.h>

#include "detr.h"
#include "utils.h"

void DETR::onnx_to_trt(nvinfer1::BuilderFlag precision,
                       std::string onnx_path,
                       std::string calibration_img_dir_path,
                       std::string calibration_table_path,
                       std::string engine_path)
{
    UniquePtr<nvinfer1::IBuilder> builder;
    UniquePtr<nvinfer1::IBuilderConfig> builderConfig;
    UniquePtr<nvinfer1::INetworkDefinition> network;
    UniquePtr<nvinfer1::IHostMemory> modelStream;
    std::unique_ptr<Int8EntropyCalibrator2> calibrator;

    builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(this->gLogger));
    builderConfig = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    if (precision == nvinfer1::BuilderFlag::kINT8)
    {
        assert(builder->platformHasFastInt8());
        std::cout << "Build INT8 engine" << std::endl;

        calibrator = std::unique_ptr<Int8EntropyCalibrator2>(
            new Int8EntropyCalibrator2(this->batchSize,
                                       this->inputW,
                                       this->inputH,
                                       calibration_img_dir_path.c_str(),
                                       calibration_table_path.c_str(),
                                       this->input_blob_name.c_str())
        );
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    assert(network != nullptr);

    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, this->gLogger));
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(this->gLogger.getReportableSeverity())))
        std::cout << "Failure while parsing ONNX file" << std::endl;

    builder->setMaxBatchSize(this->batchSize);

    builderConfig->setMaxWorkspaceSize(1 * (1 << 30));

    builderConfig->setFlag(precision);
    if (precision == nvinfer1::BuilderFlag::kINT8)
        builderConfig->setInt8Calibrator(calibrator.get());

    this->engine = UniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*(network), *(builderConfig)));
    assert(this->engine != nullptr);
    std::cout << "Build engine successfully" << std::endl;

    (modelStream) = UniquePtr<nvinfer1::IHostMemory>(this->engine->serialize());
    assert(modelStream != nullptr);

    std::ofstream p(engine_path.c_str(),
                    std::ios::binary);

    if (!p)
        assert(false);

    p.write(reinterpret_cast<const char *>(modelStream->data()),
            modelStream->size());
}

void DETR::load_engine(std::string engine_path)
{
    this->runtime = UniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->gLogger));
    assert(this->runtime != nullptr);

    std::string trtModelStream;
    size_t modelSize{0};
    std::ifstream file(engine_path.c_str(),
                       std::ios::binary);

    if (file.good())
    {
        file.seekg(0, file.end);
        modelSize = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream.resize(modelSize);
        assert(!trtModelStream.empty());
        file.read(const_cast<char*>(trtModelStream.c_str()), modelSize);
        file.close();
    }

    this->engine = UniquePtr<nvinfer1::ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize));
    assert(this->engine != nullptr);

    this->context = UniquePtr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
    assert(this->context != nullptr);

    this->inputIndex = this->engine->getBindingIndex(input_blob_name.c_str());
    this->logitsIndex = this->engine->getBindingIndex(logits_blob_name.c_str());
    this->boxesIndex = this->engine->getBindingIndex(boxes_blob_name.c_str());

    assert(this->inputIndex == 0);
    assert(this->logitsIndex == 1);
    assert(this->boxesIndex == 2);

    this->logitsSize = this->numQuery * this->numClass;
    this->boxesSize = this->numQuery * 4;

    this->inData = std::unique_ptr<float>(new float[this->batchSize * 3 * this->inputH * this->inputW]());
    this->outLogits = std::vector<float>(this->batchSize * this->logitsSize);
    this->outBoxes = std::vector<float>(this->batchSize * this->boxesSize);

    CUDA_CHECK(cudaMalloc(&this->buffers[this->inputIndex], this->batchSize * 3 * this->inputH * this->inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&this->buffers[this->logitsIndex], this->batchSize * this->logitsSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&this->buffers[this->boxesIndex], this->batchSize * this->boxesSize * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&this->stream));
}