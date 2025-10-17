import os
import torch
import torch.nn as nn
from udnet_infer import UDNetEnhancer


class UDNetONNX(nn.Module):
    def __init__(self, enhancer: UDNetEnhancer):
        super().__init__()
        self.net = enhancer.model
        self.decoder = self.net.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run decoder in inference mode (training=False).
        # Target is unused at inference, pass x for both.
        return self.decoder.forward(x, x, training=False)


if __name__ == "__main__":
    print("📂 Loading model...")
    weights_path = os.path.join("weights", "UDnet.pth")
    enhancer = UDNetEnhancer(
        weights_path=weights_path, device="cpu", gpu_mode=False
    )
    enhancer.model.eval()

    wrapper = UDNetONNX(enhancer).eval()
    print("✅ Model loaded")

    # Use a dummy input with H and W as multiples of 8
    dummy = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        wrapper,
        dummy,
        "UDnet_dynamic.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "h", 3: "w"},
            "output": {0: "batch", 2: "h", 3: "w"},
        },
        opset_version=12,
        do_constant_folding=True,
    )

    print("✅ Export finished, saved as UDnet_dynamic.onnx")
