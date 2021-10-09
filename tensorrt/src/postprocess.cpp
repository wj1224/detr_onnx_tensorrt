#include "detr.h"

void DETR::postprocess(std::vector<cv::Mat> imgs)
{
    this->batch_res = std::vector<std::vector<ObjectsInfo>>(this->batchSize);

    for (int b = 0; b < this->batchSize; b++) {
        auto &res = this->batch_res[b];
        for (int i = b * this->logitsSize; i < (b + 1) * this->logitsSize; i+= this->numClass) {
            int label = -1;
            float score = -1;
            for (int j = i; j < i + this->numClass; j++) {
                if (score < this->outLogits[j])  {
                    label = j;
                    score = this->outLogits[j];
                }
            }
            if (score > this->confThresh && (label % this->numClass != this->numClass - 1)) {
                ObjectsInfo det;
                int ind = label / this->numClass;
                label = label % this->numClass;
                float cx = this->outBoxes[ind * 4];
                float cy = this->outBoxes[ind * 4 + 1];
                float w = this->outBoxes[ind * 4 + 2];
                float h = this->outBoxes[ind * 4 + 3];
                float x1 = (cx - w / 2.0) * imgs[b].cols;
                float y1 = (cy - h / 2.0) * imgs[b].rows;
                float x2 = (cx + w / 2.0) * imgs[b].cols;
                float y2 = (cy + h / 2.0) * imgs[b].rows;
                det.conf_score = score;
                det.class_id = label;
                det.bbox[0] = x1;
                det.bbox[1] = y1;
                det.bbox[2] = x2;
                det.bbox[3] = y2;
                res.push_back(det);
            }
        }
    }
}