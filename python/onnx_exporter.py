import io
import argparse
import numpy as np
from pathlib import Path

import onnx
import onnxruntime
from onnxsim import simplify
import onnx_graphsurgeon as gs
import torch

from detr.hubconf import detr_resnet50

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/detr-r50-e632da11.pth", help="weights file path")
    parser.add_argument("--output", type=str, default="./output/detr_resnet50.onnx", help="output onnx file path")
    parser.add_argument("--batch-size", type=int, default=1, help="set batch size")
    parser.add_argument("--input-size", type=int, nargs='+', default=[800, 800], help="input size [h, w]")
    parser.add_argument("--input-blob", type=str, default="images", help="set input blob name")
    parser.add_argument("--output-blob", type=str, nargs='+', default=["pred_logits", "pred_boxes"], help="set output blob name")
    args = parser.parse_args()

    return args

@torch.no_grad()
def ort_validate(onnx_io, inputs, outputs, tolerate_small_mismatch=False):

    inputs, _ = torch.jit._flatten(inputs)
    outputs, _ = torch.jit._flatten(outputs)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    inputs = list(map(to_numpy, inputs))
    outputs = list(map(to_numpy, outputs))

    ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
    # compute onnxruntime output prediction
    ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
    ort_outs = ort_session.run(None, ort_inputs)
    for i in range(0, len(outputs)):
        try:
            torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print("tolerate small mismatch")
            else:
                raise

def onnx_change(sim_path, changed_path, batch_size):
    '''该部分代码由导师提供，解决trt inference 全是0的问题，感谢!!!
    '''
    node_configs = [(2711, 2713), # batch-size 1
        (2978, 2980)] # batch-size 4
    if batch_size == 4:
        node_number = node_configs[1]
    else:
        node_number = node_configs[0]

    graph = gs.import_onnx(onnx.load(sim_path))
    for node in graph.nodes:
        if node.name == f"Gather_{node_number[0]}":
            print(node.inputs[1])
            node.inputs[1].values = np.int64(5)
            print(node.inputs[1])
        elif node.name == f"Gather_{node_number[1]}":
            print(node.inputs[1])
            node.inputs[1].values = np.int64(5)
            print(node.inputs[1])
                
    onnx.save(gs.export_onnx(graph),changed_path)
    print("Gather node changed ONNX model saved")

if __name__ == "__main__":
    args = create_parser()

    model = detr_resnet50(pretrained=False, num_classes=91)
    state_dict = torch.load(args.checkpoint, map_location=torch.device("cuda:0"))
    model.load_state_dict(state_dict["model"])
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, args.input_size[0], args.input_size[1])
    onnxio = io.BytesIO()

    torch.onnx.export(
        model,
        dummy_input,
        onnxio,
        input_names=[args.input_blob],
        output_names=args.output_blob,
        export_params=True,
        training=False,
        opset_version=12
    )

    dir_path = Path(args.output).parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=[args.input_blob],
        output_names=args.output_blob,
        export_params=True,
        training=False,
        opset_version=12
    )
    print("ONNX model export successed")

    dummy_input = (dummy_input,)
    test_output = model(*dummy_input)
    test_output = (test_output,)
    ort_validate(onnxio, dummy_input, test_output, tolerate_small_mismatch=True)

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check successed")

    print("Simplifying ONNX model ...")
    simplified_model, check = simplify(
        onnx_model,
        input_shapes={args.input_blob: [args.batch_size, 3, args.input_size[0], args.input_size[1]]},
        dynamic_input_shape=False
    )

    onnx.save(simplified_model, args.output[:-5] + "_sim.onnx")
    print("Simplified ONNX model saved")

    print("Change ONNX model ...")
    onnx_change(args.output[:-5] + "_sim.onnx", args.output[:-5] + "_sim_changed.onnx", args.batch_size)
