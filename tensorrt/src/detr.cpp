#include "detr.h"

DETR::DETR(std::string input_blob_name,
           std::string logits_blob_name,
           std::string boxes_blob_name,
           int deviceId,
           int inputH,
           int inputW,
           int numQuery,
           int numClass,
           int batchSize,
           float confThresh) : input_blob_name(input_blob_name),
                               logits_blob_name(logits_blob_name),
                               boxes_blob_name(boxes_blob_name),
                               deviceId(deviceId),
                               inputH(inputH),
                               inputW(inputW),
                               numQuery(numQuery),
                               numClass(numClass),
                               batchSize(batchSize),
                               confThresh(confThresh)
{
    cudaSetDevice(deviceId);
    this->inData = nullptr;
    this->buffers[0] = nullptr;
    this->buffers[1] = nullptr;
    this->buffers[2] = nullptr;
    this->stream = nullptr;
}

DETR::~DETR()
{
    if (this->stream)
    {
        cudaStreamDestroy(this->stream);
        CUDA_CHECK(cudaFree(this->buffers[this->inputIndex]));
        CUDA_CHECK(cudaFree(this->buffers[this->logitsIndex]));
        CUDA_CHECK(cudaFree(this->buffers[this->boxesIndex]));
    }
}