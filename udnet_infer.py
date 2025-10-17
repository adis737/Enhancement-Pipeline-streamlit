from typing import Optional, Union, Tuple

import torch
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from model_utils.UDnet import mynet


class UDNetEnhancer:
    """
    Lightweight Python interface for UDnet inference.

    Usage:
        enhancer = UDNetEnhancer(
            weights_path="weights/UDnet.pth",
            device="cuda:0",  # or "cpu"
            gpu_mode=True,
        )
        enhanced_pil = enhancer.enhance_image(
            "./check_IQA/input/image.jpg"
        )
        enhanced_pil.save(
            "./check_IQA/output/image.jpg"
        )
    """

    def __init__(
        self,
        weights_path: str = "weights/UDnet.pth",
        device: str = "cuda:0",
        gpu_mode: bool = True,
    ) -> None:
        self.device_str = device
        self.device = torch.device(device)
        self.gpu_mode = (
            gpu_mode and torch.cuda.is_available() and
            self.device.type == "cuda"
        )

        # Construct minimal opt namespace expected by mynet
        class _Opt:
            def __init__(self, device_str: str) -> None:
                self.device = device_str

        self.model = mynet(_Opt(self.device_str))
        state = torch.load(
            weights_path, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(state, strict=False)
        if self.gpu_mode:
            self.model = self.model.cuda(self.device)
        self.model.eval()

        # no gradients during inference
        torch.set_grad_enabled(False)

    @staticmethod
    def _load_image(
        img: Union[str, Image.Image, np.ndarray]
    ) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                # Normalize and convert to uint8 if necessary
                arr = np.clip(img, 0, 255).astype(np.uint8)
            else:
                arr = img
            return Image.fromarray(arr).convert("RGB")
        raise TypeError("img must be a file path, PIL.Image, or numpy.ndarray")

    @staticmethod
    def _center_crop_to_multiple(
        pil_img: Image.Image, multiple: int = 8
    ) -> Image.Image:
        w, h = pil_img.size
        new_w = w - (w % multiple)
        new_h = h - (h % multiple)
        if new_w == w and new_h == h:
            return pil_img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        return pil_img.crop((left, top, right, bottom))

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        # Convert to CHW float tensor in [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(chw)

    @staticmethod
    def _tensor_to_pil(
        t: torch.Tensor
    ) -> Image.Image:
        # Expect CHW in [0,1]
        t = t.detach().cpu().clamp(0, 1)
        arr = (t.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _apply_gray_world(pil_img: Image.Image) -> Image.Image:
        # Gray-world white balance: scale channels so their averages match
        arr = np.asarray(pil_img).astype(np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return pil_img
        mean = arr.reshape(-1, 3).mean(axis=0) + 1e-6
        gray = mean.mean()
        scale = gray / mean
        balanced = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(balanced)

    def enhance_image(
        self,
        img: Union[str, Image.Image, np.ndarray],
        return_size: Optional[Tuple[int, int]] = None,
        max_side: Optional[int] = None,
        neutralize_cast: bool = False,
        saturation: float = 1.0,
    ) -> Image.Image:
        """
        Enhance a single image and return a PIL.Image in RGB.

        Args:
            img: path, PIL.Image, or numpy array (HWC, RGB).
            return_size: if provided, resize output back to (width, height).
        """
        pil = self._load_image(img)
        original_size = pil.size

        # Optional downscale for memory safety
        pil_scaled = pil
        if max_side is not None and max_side > 0:
            w, h = pil.size
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                new_w = max(8, int(round(w * scale)))
                new_h = max(8, int(round(h * scale)))
                pil_scaled = pil.resize((new_w, new_h), resample=Image.BICUBIC)

        pil_proc = self._center_crop_to_multiple(pil_scaled, multiple=8)
        tensor = self._pil_to_tensor(pil_proc).unsqueeze(0)  # [1, C, H, W]

        if self.gpu_mode:
            tensor = tensor.cuda(self.device)

        # The model expects both Input and label; for inference we pass
        # Input for both
        try:
            self.model.forward(tensor, tensor, training=False)
            pred = self.model.sample(testing=True)  # [1, C, H, W]
        except RuntimeError as e:
            # Fallback to CPU on CUDA OOM
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                self.model = self.model.to("cpu")
                self.gpu_mode = False
                tensor_cpu = tensor.detach().cpu()
                self.model.forward(tensor_cpu, tensor_cpu, training=False)
                pred = self.model.sample(testing=True)
            else:
                raise

        out_pil = self._tensor_to_pil(pred.squeeze(0))

        # Resize back to original size by default, unless caller requests a
        # specific size
        if return_size is not None:
            target_w, target_h = return_size
            out_pil = out_pil.resize(
                (target_w, target_h), resample=Image.BICUBIC
            )
        else:
            out_pil = out_pil.resize(original_size, resample=Image.BICUBIC)

        # Optional post-processing to mitigate red boost
        if neutralize_cast:
            out_pil = self._apply_gray_world(out_pil)
        if abs(saturation - 1.0) > 1e-3:
            try:
                out_pil = ImageEnhance.Color(out_pil).enhance(
                    max(0.0, saturation)
                )
            except Exception:
                pass

        return out_pil


def enhance_image(
    img: Union[str, Image.Image, np.ndarray],
    weights_path: str = "weights/UDnet.pth",
    device: str = "cuda:0",
    gpu_mode: bool = True,
    max_side: Optional[int] = None,
    neutralize_cast: bool = False,
    saturation: float = 1.0,
) -> Image.Image:
    """
    Convenience function: load model, enhance one image, return a PIL.Image.
    Prefer UDNetEnhancer if calling repeatedly to avoid reloading weights.
    """
    enhancer = UDNetEnhancer(
        weights_path=weights_path,
        device=device,
        gpu_mode=gpu_mode,
    )
    return enhancer.enhance_image(
        img,
        max_side=max_side,
        neutralize_cast=neutralize_cast,
        saturation=saturation,
    )
