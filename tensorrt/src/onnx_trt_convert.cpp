#include <glib.h>
#include "detr.h"

int main(int argc, char **argv)
{
    int deviceId = 0;
    int inputH = 800;
    int inputW = 800;
    int numQuery = 100;
    int numClass = 91 + 1; // coco 91 classes + no object
    int batchSize = 1;

    int precision = 7;

    gchar* onnx_path = "";
    gchar* calibration_img_dir_path = "";
    gchar* calibration_table_path = "";
    gchar* engine_path = "";
    gchar* input_blob_name = "images";
    gchar* output_blob_name[2] = {"pred_logits", "pred_boxes"};

    GOptionContext* optCtx = NULL;
    GError* error = NULL;

    GOptionEntry entries[] = {
        {"device", 'd', 0, G_OPTION_ARG_INT, &deviceId, "GPU device id (default: 0)", NULL},
        {"input_h", 'h', 0, G_OPTION_ARG_INT, &inputH, "input image height (default: 800)", NULL},
        {"input_w", 'w', 0, G_OPTION_ARG_INT, &inputW, "input image width (default: 800)", NULL},
        {"num_query", 'q', 0, G_OPTION_ARG_INT, &numQuery, "the maximum number of objects (default: 100)", NULL},
        {"num_class", 'c', 0, G_OPTION_ARG_INT, &numClass, "number of classes (default: 92 (91 + 1))", NULL},
        {"batch_size", 'b', 0, G_OPTION_ARG_INT, &batchSize, "batch size (default: 1)", NULL},
        {"precision", 'p', 0, G_OPTION_ARG_INT, &precision, "7(FP32), 0(FP16), 1(INT8) (default: 7)", NULL},
        {"onnx_path", 'x', 0, G_OPTION_ARG_STRING, &onnx_path, "input onnx path (default: empty string)", NULL},
        {"calib_img_dir", 'm', 0, G_OPTION_ARG_STRING, &calibration_img_dir_path, "sample images for int8 calibration (default: empty string)", NULL},
        {"calib_table", 't', 0, G_OPTION_ARG_STRING, &calibration_table_path, "output calibration table path (default: empty string)", NULL},
        {"engine", 'e', 0, G_OPTION_ARG_STRING, &engine_path, "output engine path (default: empty string)", NULL},
        {"input_blob", 'i', 0, G_OPTION_ARG_STRING, &input_blob_name, "input blob name (default: images)", NULL},
        {"output_blob", 'o', 0, G_OPTION_ARG_STRING_ARRAY, &output_blob_name, "output blob name (default: {pred_logits, pred_boxes}", NULL},
        {NULL} };

    optCtx = g_option_context_new ("onnx_to_tensorrt");

    g_option_context_add_main_entries (optCtx, entries, NULL);

    if (!g_option_context_parse (optCtx, &argc, &argv, &error)) {
            g_print ("option parsing failed: %s\n", error->message);
            g_option_context_free (optCtx);
            exit (1);
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
                    batchSize);

    detr->onnx_to_trt((nvinfer1::BuilderFlag)precision,
                       onnx_path,
                       calibration_img_dir_path,
                       calibration_table_path,
                       engine_path);

    delete detr;

    return 0;
}