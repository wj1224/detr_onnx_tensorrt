#include "detr.h"
#include "utils.h"

float *DETR::infer(std::vector<cv::Mat> imgs)
{
    auto data = this->inData.get();
        
    int i = 0;
    for (int b = 0; b < this->batchSize; b++)
    {
        cv::Mat pr_img = preprocess(imgs[b], this->inputW, this->inputH);
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < pr_img.rows; h++) {
                for (int w = 0; w < pr_img.cols; w++) {
                    data[b * 3 * this->inputH * this->inputW + c * pr_img.rows * pr_img.cols + h * pr_img.cols + w] = pr_img.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(this->buffers[0],
                               this->inData.get(),
                               this->batchSize * 3 * this->inputH * this->inputW * sizeof(float),
                               cudaMemcpyHostToDevice,
                               this->stream));

    context->enqueue(this->batchSize,
                     this->buffers,
                     this->stream,
                     nullptr);

    CUDA_CHECK(cudaMemcpyAsync(this->outLogits.data(),
                               this->buffers[1],
                               this->batchSize * this->logitsSize * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               this->stream));
    CUDA_CHECK(cudaMemcpyAsync(this->outBoxes.data(),
                               this->buffers[2],
                               this->batchSize * this->boxesSize * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               this->stream));

    cudaStreamSynchronize(this->stream);

    return nullptr;
}