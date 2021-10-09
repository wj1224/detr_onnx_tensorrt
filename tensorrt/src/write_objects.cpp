#include "detr.h"

void DETR::write_objects(std::vector<cv::Mat> imgs, std::vector<std::string> out_img_lists)
{
    for (int b = 0; b < this->batchSize; b++) {
        auto &res = this->batch_res[b];
        cv::Mat img = imgs[b];
        for (size_t j = 0; j < res.size(); j++) {
            float x1 = res[j].bbox[0];
            float y1 = res[j].bbox[1];
            float x2 = res[j].bbox[2];
            float y2 = res[j].bbox[3];
            float conf_score = res[j].conf_score;
            int class_id = res[j].class_id;

            cv::Rect r(x1, y1, x2 - x1, y2 - y1);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string(class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite(out_img_lists[b], img);
    }
}