#include <glib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <map>
#include <tuple>
#include "detr.h"
#include "utils.h"

int main(int argc, char **argv)
{
    gchar *image_dir = "";
    gchar *image_out_dir = "";

    int deviceId = 0;
    int inputH = 800;
    int inputW = 800;
    int numQuery = 100;
    int numClass = 91 + 1; // coco 91 classes + no object
    int batchSize = 1;
    float confThresh = 0.25;

    gchar* engine_path = NULL;
    gchar* input_blob_name = "images";
    gchar* output_blob_name[2] = {"pred_logits", "pred_boxes"};

    GOptionContext *optCtx = NULL;
    GError *error = NULL;

    GOptionEntry entries[] = {
        {"device", 'd', 0, G_OPTION_ARG_INT, &deviceId, "GPU device id (default: 0)", NULL},
        {"input_h", 'h', 0, G_OPTION_ARG_INT, &inputH, "input image height (default: 800)", NULL},
        {"input_w", 'w', 0, G_OPTION_ARG_INT, &inputW, "input image width (default: 800)", NULL},
        {"num_query", 'q', 0, G_OPTION_ARG_INT, &numQuery, "the maximum number of objects (default: 100)", NULL},
        {"num_class", 'c', 0, G_OPTION_ARG_INT, &numClass, "number of classes (default: 92 (91 + 1))", NULL},
        {"batch_size", 'b', 0, G_OPTION_ARG_INT, &batchSize, "batch size (default: 1)", NULL},
        {"conf_th", 'f', 0, G_OPTION_ARG_DOUBLE, &confThresh, "confidence score threshold (default: 0.25)", NULL},
        {"image_dir", 'm', 0, G_OPTION_ARG_STRING, &image_dir, "input image dir(default: empty string)", NULL},
        {"image_out_dir", 't', 0, G_OPTION_ARG_STRING, &image_out_dir, "output image dir(default: empty string)", NULL},
        {"engine", 'e', 0, G_OPTION_ARG_STRING, &engine_path, "input engine path (default: empty string)", NULL},
        {"input_blob", 'i', 0, G_OPTION_ARG_STRING, &input_blob_name, "input blob name (default: images)", NULL},
        {"output_blob", 'o', 0, G_OPTION_ARG_STRING_ARRAY, &output_blob_name, "output blob name (default: {pred_logits, pred_boxes}", NULL},
        {NULL} };
    
    optCtx = g_option_context_new("detect");

    g_option_context_add_main_entries(optCtx, entries, NULL);
    
    if (!g_option_context_parse(optCtx, &argc, &argv, &error))
    {
        g_print("option parsing failed: %s\n", error->message);
        g_option_context_free(optCtx);
        exit(1);
    }

    DETR *detr;
    
    detr = new DETR(input_blob_name,
                    output_blob_name[0],
                    output_blob_name[1],
                    deviceId,
                    inputH,
                    inputW,
                    numQuery,
                    numClass,
                    batchSize,
                    confThresh);

    std::cout << "Load engine ..." << std::endl;
    detr->load_engine(engine_path);

    std::vector<std::string> file_names;
    std::vector<std::string> img_lists;

    if (read_files_in_dir(image_dir, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    std::vector<cv::Mat> imgs;
    std::vector<std::string> image_out_path;
    cv::Mat img;

    for (int i = 0; i < file_names.size(); i++)
    {
        img_lists.push_back(file_names[i]);
        if (img_lists.size() < batchSize) // size of file_names must be multiple of batchSize else remains will be throw away.
            continue;
        for (int j = 0; j < img_lists.size(); j++)
        {
            if (image_dir)
            {
                img = cv::imread(std::string(image_dir) + "/" + img_lists[j]);
                imgs.push_back(img);
            }

            if (image_out_dir) {
                if (image_dir) {
                    image_out_path.push_back(std::string(image_out_dir) + "/" + img_lists[j]);
                }
            }
        }
            
        auto start = std::chrono::system_clock::now();
        detr->infer(imgs);
        auto end_infer = std::chrono::system_clock::now();
        std::cout << "Inference time per batch: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start).count() << "ms" << std::endl;

        detr->postprocess(imgs);
 
        detr->write_objects(imgs, image_out_path);
        
        img_lists.clear();
        image_out_path.clear();
        imgs.clear();

    }
    std::cout << "Inference ended successfully." << std::endl;

    delete detr;

    return 0;
}
