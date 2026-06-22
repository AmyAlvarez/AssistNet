import os
import torch
import torch.nn as nn

ROOT       = r"C:\Users\Your_path\AssistNet\AssistNet_v2"
MODELS_DIR = os.path.join(ROOT, "models")
ONNX_PATH  = os.path.join(MODELS_DIR, "assistnet_v2.onnx")

# device info
device = torch.device("cpu")
print("Exporting on CPU for ONNX compatibility")


class AssistNetV2(nn.Module):
    def __init__(self):
        super(AssistNetV2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# load weights
model = AssistNetV2().to(device)
model.load_state_dict(
    torch.load(
        os.path.join(MODELS_DIR, "assistnet_v2_best.pth"),
        map_location=device,
    )
)


model.eval()
print("Model loaded and set to eval mode")

dummy_input = torch.randn(1, 3, 150, 150)

# actual export 
print(f"\nExporting to ONNX: {ONNX_PATH}")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    dynamo=False,    # force legacy exporter — keeps weights in single file
)

print("Export complete")

# verify onnx file
try:
    import onnx
    model_onnx = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed!")
    print(f"\nModel inputs:  {[i.name for i in model_onnx.graph.input]}")
    print(f"Model outputs: {[o.name for o in model_onnx.graph.output]}")
except ImportError:
    print("onnx package not installed skipping check.")
    print("Install with: pip install onnx")

# sanity check
try:
    import onnxruntime as ort
    import numpy as np

    # PyTorch output
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    # ONNX Runtime output
    ort_session = ort.InferenceSession(ONNX_PATH)
    ort_output  = ort_session.run(
        None,
        {"input": dummy_input.numpy()}
    )[0]

    max_diff = np.abs(pt_output - ort_output).max()
    print(f"\nSanity check max output difference PyTorch vs ONNX: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("Outputs match export is sound")
    else:
        print("WARNING: outputs differ check the export")

except ImportError:
    print("\nonnxruntime not installed skipping sanity check.")
    print("Install with: pip install onnxruntime")

print(f"\nONNX model saved to: {ONNX_PATH}")
