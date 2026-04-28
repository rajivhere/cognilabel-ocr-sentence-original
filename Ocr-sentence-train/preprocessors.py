import cv2
import numpy as np

class BinarizeNormalize:
    def __init__(
        self,
        method="adaptive_gaussian",
        mode="soft",
        block_size=21,
        C=10,
        threshold=160,
        keep_3ch=True,
    ):
        self.method = method
        self.mode = mode
        self.block_size = self._ensure_odd(block_size)
        self.C = float(C)
        self.threshold = int(max(0, min(255, threshold)))
        self.keep_3ch = bool(keep_3ch)

    def _ensure_odd(self, v):
        v = int(v)
        if v < 3:
            v = 3
        return v if v % 2 == 1 else v + 1

    def _to_gray(self, image):
        if len(image.shape) == 2:
            return image, False
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), True

    def _apply_hard(self, gray):
        if self.method == "otsu":
            _, out = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return out

        if self.method == "adaptive_mean":
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                self.C,
            )

        if self.method == "adaptive_gaussian":
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                self.C,
            )

        # fallback: fixed threshold
        _, out = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return out

    def _apply_soft(self, gray):
        """
        Soft binarization:
        - compute a binary guide
        - push background toward white
        - push foreground toward black
        - preserve some grayscale detail
        """
        binary = self._apply_hard(gray)

        gray_f = gray.astype(np.float32)
        mask_bg = binary == 255
        mask_fg = ~mask_bg

        out = gray_f.copy()

        # push background brighter
        out[mask_bg] = np.minimum(255.0, 220.0 + (gray_f[mask_bg] - 128.0) * 0.35)

        # push foreground darker while preserving detail
        out[mask_fg] = np.maximum(0.0, gray_f[mask_fg] * 0.35)

        return out.astype(np.uint8)

    def __call__(self, data):
        image, label = data
        if image is None:
            return data

        gray, was_color = self._to_gray(image)

        if self.mode == "hard":
            out = self._apply_hard(gray)
        else:
            out = self._apply_soft(gray)

        if self.keep_3ch:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        return out, label
    
class ToGrayscale:
    def __init__(self, strength=1.0, keep_3ch=True):
        self.strength = float(max(0.0, min(1.0, strength)))
        self.keep_3ch = bool(keep_3ch)

    def __call__(self, data):
        image, label = data
        if image is None:
            return data

        if len(image.shape) == 2:
            gray = image
            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color = image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.strength >= 1.0:
            out = gray
        elif self.strength <= 0.0:
            out = color
        else:
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(color, 1.0 - self.strength, gray3, self.strength, 0)

        if self.keep_3ch:
            if len(out.shape) == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            return out, label

        if len(out.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        return out, label


class NormalizePolarity:
    def __init__(self, target="dark_on_light", mode="auto"):
        self.target = target
        self.mode = mode

    def __call__(self, data):
        image, label = data
        if image is None:
            return data

        is_color = len(image.shape) == 3
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image
        out = gray

        if self.mode == "off":
            out = gray
        else:
            mean_val = float(np.mean(gray))
            looks_dark_background = mean_val < 127

            should_invert = False

            if self.target == "dark_on_light" and looks_dark_background:
                should_invert = True

            if self.target == "light_on_dark" and not looks_dark_background:
                should_invert = True

            if should_invert:
                out = 255 - gray

        if is_color:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        return out, label


class NormalizeContrast:
    def __init__(self, method="clahe", clip_limit=2.0, tile_grid_size=(8, 8)):
        self.method = method
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        image, label = data
        if image is None:
            return data

        is_color = len(image.shape) == 3
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

        if self.method == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
            out = clahe.apply(gray)
        elif self.method == "equalize":
            out = cv2.equalizeHist(gray)
        elif self.method == "minmax":
            out = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            out = gray

        if is_color:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        return out, label