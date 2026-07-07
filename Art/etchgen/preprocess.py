"""Image loading and tone conditioning. Everything downstream works on
float32 brightness in [0, 1] (1 = white) at the working resolution."""

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path):
    """Path -> RGB uint8 array, EXIF-rotated."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))


def decode_image(data):
    """Raw bytes (upload) -> RGB uint8 array, EXIF-rotated."""
    import io
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))


def preprocess(rgb, p):
    """RGB uint8 -> dict(gray, tone) float32 brightness maps in [0,1].

    gray: full-detail conditioned image (edge detection input)
    tone: lightly blurred copy (shading density input)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    scale = p["detail"] / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(2, round(w * scale)), max(2, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    elif scale > 1.0:
        # never upsample: "detail" is a ceiling, not a target
        pass

    if p["clahe"] > 0:
        clahe = cv2.createCLAHE(clipLimit=float(p["clahe"]), tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if p["denoise"] > 0:
        d = float(p["denoise"])
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=60 * d, sigmaSpace=4 + 6 * d)

    g = gray.astype(np.float32) / 255.0
    if p["invert"]:
        g = 1.0 - g

    lo, hi = float(p["black_point"]), float(p["white_point"])
    if hi - lo < 0.05:
        hi = lo + 0.05
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    gamma = float(p["gamma"])
    if abs(gamma - 1.0) > 1e-3:
        g = np.power(g, 1.0 / gamma, dtype=np.float32)

    tone = cv2.GaussianBlur(g, (0, 0), 2.0)
    return {"gray": g.astype(np.float32), "tone": tone.astype(np.float32)}
