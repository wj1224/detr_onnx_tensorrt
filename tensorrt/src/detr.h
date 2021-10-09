#ifndef __DETR_H__
#define __DETR_H__

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "logging.h"
#include "cuda_utils.h"
#include "calibrator.h"

struct AutoDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
            obj->destroy();
    }
};

struct alignas(float) ObjectsInfo
{
    float bbox[4];
    float conf_score;
    int class_id;
};

class DETR
{
    template <typename T>
    using UniquePtr = std::unique_ptr<T, AutoDeleter>;

public:
    DETR(std::string input_blob_name,
         std::string logits_blob_name,
         std::string boxes_blob_name,
         int deviceId = 0,
         int inputH = 800,  
         int inputW = 800,
         int numQuery = 100,
         int numClass = 91,
         int batchSize = 1,
         float confThresh = 0.25);

    ~DETR();

    void onnx_to_trt(nvinfer1::BuilderFlag precision,
                     std::string onnx_path,
                     std::string calibration_img_dir_path,
                     std::string calibration_table_path,
                     std::string engine_path);

    void load_engine(std::string engine_path);

    float *infer(std::vector<cv::Mat> imgs);

    void postprocess(std::vector<cv::Mat> imgs);

    void write_objects(std::vector<cv::Mat> imgs, std::vector<std::string> out_img_lists);

private:
    const std::string input_blob_name;
    const std::string logits_blob_name;
    const std::string boxes_blob_name;
    const int deviceId;
    const int inputH;
    const int inputW;
    const int numQuery;
    const int numClass;
    const int batchSize;
    const float confThresh;

    UniquePtr<nvinfer1::ICudaEngine> engine;
    UniquePtr<nvinfer1::IRuntime> runtime;
    UniquePtr<nvinfer1::IExecutionContext> context;

    Logger gLogger;

    std::unique_ptr<float> inData;
    std::vector<float> outLogits;
    std::vector<float> outBoxes;
    int logitsSize;
    int boxesSize;

    void *buffers[3];
    int inputIndex;
    int logitsIndex;
    int boxesIndex;
    cudaStream_t stream;

    std::vector<std::vector<ObjectsInfo>> batch_res;
};

#endif