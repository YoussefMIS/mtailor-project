import torch
import torch.nn as nn
from pytorch_model import Classifier, BasicBlock  # Explicit imports instead of star imports

def converter(model: nn.Module, input_shape: tuple, output_path: str):
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'], opset_version=11)


if __name__ == "__main__":
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("pytorch_model_weights.pth"))
    converter(mtailor, (3, 224, 224), "mtailor.onnx")