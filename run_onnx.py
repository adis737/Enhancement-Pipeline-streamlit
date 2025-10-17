import sys
import numpy as np
from PIL import Image


def to_multiple_of_8(w, h):
    return w - (w % 8), h - (h % 8)


def preprocess(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, 0)


def postprocess(out):
    out = np.clip(out[0].transpose(1, 2, 0), 0, 1)
    return Image.fromarray((out * 255).round().astype(np.uint8))


def main():
    try:
        import onnxruntime as ort
    except Exception:
        raise RuntimeError(
            "Please install onnxruntime: pip install onnxruntime"
        )

    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python run_onnx.py <model.onnx> <input.jpg> <output.jpg>"
        )

    onnx_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    pil = Image.open(in_path).convert("RGB")
    w, h = pil.size
    new_w, new_h = to_multiple_of_8(w, h)
    pil_resized = pil.resize((new_w, new_h), Image.BICUBIC)

    inp = preprocess(pil_resized).astype(np.float32)
    out = sess.run([out_name], {inp_name: inp})[0]
    out_img = postprocess(out).resize((w, h), Image.BICUBIC)
    out_img.save(out_path)


if __name__ == "__main__":
    main()
