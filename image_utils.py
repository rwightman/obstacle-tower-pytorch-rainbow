import numpy as np
import random

GRAY_COEF = np.array([0.299, 0.587, 0.114], dtype=np.float32)
TYIQ = np.array(
    [[0.299, 0.587, 0.114],
     [0.596, -0.274, -0.321],
     [0.211, -0.523, 0.311]], dtype=np.float32)

ITYIQ = np.array(
    [[1.0, 0.956, 0.621],
     [1.0, -0.272, -0.647],
     [1.0, -1.107, 1.705]], dtype=np.float32)


def augment_brightness(img, alpha):
    return np.clip(img * alpha, 0, 255.)


def augment_saturation(img, alpha):
    assert len(img.shape) == 3 and img.shape[2] == 3
    gray = img * GRAY_COEF
    gray = np.sum(gray, keepdims=True, axis=2, dtype=np.float32)
    gray *= (1.0 - alpha)
    img = img * alpha + gray
    img = np.clip(img, 0, 255.)
    return img


def augment_contrast(img, alpha):
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
    else:
        gray = img * GRAY_COEF
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray, dtype=np.float32)
    img = img * alpha + gray
    img = np.clip(img, 0, 255.)
    return img


def augment_hue(img, alpha):
    assert len(img.shape) == 3 and img.shape[2] == 3
    u = np.cos(alpha * np.pi)
    w = np.sin(alpha * np.pi)
    bt = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, u, -w],
         [0.0, w, u]], dtype=np.float32)
    img = np.dot(img, np.dot(np.dot(ITYIQ, bt), TYIQ).T)
    img = np.clip(img, 0, 255.)
    return img


def random_augment_color(img, hue=0.4, sat=0.4, bright=0.4, contrast=0.4, return_fn=False):
    assert len(img.shape) == 3 and img.shape[2] >= 3

    img = img.astype(np.float32)

    fns = []
    if bright:
        random_bright = 1.0 + random.uniform(-bright, bright)
        fns.append(lambda img: augment_brightness(img, random_bright))
    if sat:
        random_sat = 1.0 + random.uniform(-sat, sat)
        fns.append(lambda img: augment_saturation(img, random_sat))
    if contrast:
        random_contrast = 1.0 + random.uniform(-contrast, contrast)
        fns.append(lambda img: augment_contrast(img, random_contrast))
    if hue:
        random_hue = random.uniform(-hue, hue)
        fns.append(lambda img: augment_hue(img, random_hue))
    random.shuffle(fns)

    if return_fn:
        return fns

    for f in fns:
        img = f(img)

    return img.astype(np.uint8)


def apply_augment_fns(img, fns, out_type=np.float32):
    img = img.astype(np.float32)
    for f in fns:
        img = f(img)
    if out_type != img.dtype:
        return img.astype(out_type)
    else:
        return img


def random_augment_grayscale(img, bright=0.4, contrast=0.4):
    assert len(img.shape) == 2 or img.shape[2] == 1
    img = img.astype(np.float32)
    fns = []
    if bright:
        random_bright = 1.0 + random.uniform(-bright, bright)
        fns.append(lambda img: augment_brightness(img, random_bright))
    if contrast:
        random_contrast = 1.0 + random.uniform(-contrast, contrast)
        fns.append(lambda img: augment_contrast(img, random_contrast))

    random.shuffle(fns)
    for f in fns:
        img = f(img)

    return img.astype(np.uint8)
