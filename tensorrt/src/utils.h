#ifndef __UTILS_H__
#define __UTILS_H__

#include <numeric>
#include <dirent.h>
#include <opencv2/opencv.hpp>

static inline cv::Mat preprocess(cv::Mat& img, int input_w, int input_h) {
    cv::Mat out(input_h, input_w, CV_8UC3);

    cv::resize(img, out, cv::Size(input_w, input_h), cv::INTER_LINEAR);
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32FC3, 1 / 255.0);
    out = (out - cv::Scalar(0.485, 0.456, 0.406)) / cv::Scalar(0.229, 0.224, 0.225);

    return out;
}

static std::string get_file_extension(const std::string& FileName)
{
    if(FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".")+1);
    return "";
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cfn(p_file->d_name);

            std::string ext = get_file_extension(cfn);
            if(ext == "jpg" || ext == "jpeg" || ext == "png") {
                file_names.push_back(cfn);
            } 
        }
    }

    closedir(p_dir);
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

#endif