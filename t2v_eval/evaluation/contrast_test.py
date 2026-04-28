"""
Accessibility metric: background-text colour contrast (WCAG 2.x).
"""

import math

import numpy as np
import pandas as pd
import pytesseract
import matplotlib.colors
from PIL import Image
from sklearn.cluster import KMeans


class TextData:
    def __init__(self, text: str, font_size: int,
                 text_color, contrast_ratio: float, background_color):
        self.text             = text
        self.font_size        = font_size
        self.text_color       = text_color
        self.contrast_ratio   = contrast_ratio
        self.text_background  = background_color


class ContrastTest:
    """
    Test WCAG colour-contrast compliance for all text regions in an image.

    Parameters
    ----------
    image_path : str
        Path to the image file to analyse.
    """

    def __init__(self, image_path: str = ""):
        self.image_path = image_path
        self.image_pil  = Image.open(image_path).convert("RGB")
        self.image_np   = np.array(self.image_pil)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _join_non_empty(self, series) -> str:
        return " ".join(v.strip() for v in series if v.strip())

    def _get_image_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return non-grey pixels in the bounding box as an (N, 3) array."""
        region = self.image_np[y:y + h, x:x + w, :]
        r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
        non_grey = ~((r == g) & (g == b))
        return region[non_grey]

    def _extract_foreground_color(self, image_region: np.ndarray,
                                   use_rgb: bool = False):
        """
        K-means segment image_region into background and foreground.

        Returns
        -------
        (foreground_color, background_color)
            RGB tuples when *use_rgb* is True, hex strings otherwise.
        """
        pixels = image_region.reshape(-1, 3)
        one_color = False
        try:
            km = KMeans(n_clusters=2, random_state=0).fit(pixels)
        except Exception:
            km = KMeans(n_clusters=1, random_state=0).fit(pixels)
            one_color = True

        bg_idx  = np.argmax(np.bincount(km.labels_))
        fg_idx  = 1 - bg_idx
        bg_rgb  = km.cluster_centers_[bg_idx].astype(int)
        fg_rgb  = km.cluster_centers_[fg_idx].astype(int) if not one_color else bg_rgb

        if use_rgb:
            return tuple(fg_rgb), tuple(bg_rgb)

        def _to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        return _to_hex(fg_rgb), _to_hex(bg_rgb)

    def _extract_background_color(self) -> str:
        """Return the mean image colour as a hex string."""
        bg = np.mean(self.image_np, axis=(0, 1))
        return "#{:02x}{:02x}{:02x}".format(int(bg[0]), int(bg[1]), int(bg[2]))

    @staticmethod
    def _luminance(color, use_rgb: bool) -> float:
        """Relative luminance as defined by WCAG 2.x."""
        if use_rgb:
            r, g, b = color[0] / 255, color[1] / 255, color[2] / 255
        else:
            r, g, b = matplotlib.colors.to_rgb(color)

        def _linearise(c):
            return c / 12.92 if c <= 0.03928 else math.pow((c + 0.055) / 1.055, 2.4)

        return 0.2126 * _linearise(r) + 0.7152 * _linearise(g) + 0.0722 * _linearise(b)

    def _contrast_ratio(self, text_color, bg_color, use_rgb: bool) -> float:
        lum_t = self._luminance(text_color, use_rgb)
        lum_b = self._luminance(bg_color,   use_rgb)
        lighter, darker = max(lum_t, lum_b), min(lum_t, lum_b)
        return (lighter + 0.05) / (darker + 0.05)

    def _region_contrast(self, image_region: np.ndarray, background_color,
                          use_clusters: bool = True, use_rgb: bool = False):
        if use_clusters:
            text_color, background_color = self._extract_foreground_color(
                image_region, use_rgb=use_rgb)
        else:
            mean = np.mean(image_region, axis=0)
            text_color = ("#{:02x}{:02x}{:02x}".format(*mean.astype(int))
                          if not use_rgb else tuple(mean.astype(int)))
        ratio = self._contrast_ratio(text_color, background_color, use_rgb)
        return ratio, text_color, background_color

    def _extract_text_colors(self, use_clusters: bool = True,
                              use_rgb: bool = False, opt: str = "2"):
        data = pd.DataFrame(
            pytesseract.image_to_data(self.image_pil,
                                      output_type=pytesseract.Output.DICT)
        )
        if opt in ("2", "3"):
            data = data[data["conf"] > 30]
        if opt in ("1", "3"):
            data = data[data["text"].str.strip() != ""]

        bg_color    = self._extract_background_color()
        text_list   = []
        total_count = 0

        for _, row in data.iterrows():
            x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
            region = self._get_image_region(x, y, w, h)
            total_count += 1
            if len(region) > 0:
                ratio, text_color, bg = self._region_contrast(
                    region, bg_color, use_clusters=use_clusters, use_rgb=use_rgb)
                text_list.append(TextData(
                    text=str(row["text"]),
                    font_size=int(h),
                    text_color=text_color,
                    contrast_ratio=ratio,
                    background_color=bg,
                ))
        return text_list, total_count

    def _extract_block_colors(self, use_clusters: bool = True,
                               use_rgb: bool = False, opt: str = "2"):
        data = pd.DataFrame(
            pytesseract.image_to_data(self.image_pil,
                                      output_type=pytesseract.Output.DICT)
        )
        if opt in ("2", "3"):
            data = data[data["conf"] > 30]
        if opt in ("1", "3"):
            data = data[data["text"].str.strip().apply(len) > 0]

        bg_color    = self._extract_background_color()
        total_count = len(data)
        text_list   = []

        all_blocks = data[data["block_num"].isin(data["block_num"])]
        for _, group in all_blocks.groupby("block_num"):
            group = group.sort_values("word_num")
            x, y = int(group["left"].min()),  int(group["top"].min())
            w, h = int(group["width"].max()), int(group["height"].max())
            region = self._get_image_region(x, y, w, h)
            if len(region) > 0:
                ratio, text_color, bg = self._region_contrast(
                    region, bg_color, use_clusters=use_clusters, use_rgb=use_rgb)
                text_list.append(TextData(
                    text=self._join_non_empty(group["text"]),
                    font_size=h,
                    text_color=text_color,
                    contrast_ratio=ratio,
                    background_color=bg,
                ))
        return text_list, total_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_test(self, verbose: bool = False,
                 use_clusters: bool = True,
                 use_blocks: bool = False,
                 use_rgb: bool = False,
                 opt: str = "2") -> tuple[dict, list]:
        """
        Run the WCAG contrast test on the image.

        Returns
        -------
        (summary_results, info_list)
            *summary_results* contains ``"score"`` (fraction of compliant text
            elements) and detailed statistics.
            *info_list* contains per-element dicts.
        """
        if use_blocks:
            text_items, total_count = self._extract_block_colors(
                use_clusters=use_clusters, use_rgb=use_rgb, opt=opt)
        else:
            text_items, total_count = self._extract_text_colors(
                use_clusters=use_clusters, use_rgb=use_rgb, opt=opt)

        self.image_pil.close()

        unsuccessful = 0
        info_list       = []
        labels          = []
        contrast_ratios = []

        for item in text_items:
            label = 0
            if item.font_size < 14 and item.contrast_ratio < 4.5:
                unsuccessful += 1
                label = 1
            elif item.contrast_ratio < 3:
                unsuccessful += 1
                label = 2

            info_list.append({
                "size":              item.font_size,
                "contrast_ratio":    item.contrast_ratio,
                "color":             item.text_color,
                "text_background":   item.text_background,
                "label":             label,
                "total_text_number": total_count,
            })
            labels.append(label)
            contrast_ratios.append(item.contrast_ratio)

            if verbose:
                print(f"  size={item.font_size} cr={item.contrast_ratio:.2f} "
                      f"label={label} text='{item.text}'")

        try:
            score = 1.0 - unsuccessful / total_count
        except ZeroDivisionError:
            score = 1.0

        arr_cr     = np.array(contrast_ratios)
        arr_labels = np.array(labels)
        summary = {
            "score":            score,
            "non_black_success": int(np.count_nonzero(arr_labels == 0)),
            "failed_small":      int(np.count_nonzero(arr_labels == 1)),
            "failed_large":      int(np.count_nonzero(arr_labels == 2)),
            "text_number":       total_count,
            "max":  float(np.max(arr_cr))              if len(arr_cr) else 0.0,
            "min":  float(np.min(arr_cr))              if len(arr_cr) else 0.0,
            "avg":  float(np.mean(arr_cr))             if len(arr_cr) else 0.0,
            "q1":   float(np.percentile(arr_cr, 25))   if len(arr_cr) else 0.0,
            "q2":   float(np.percentile(arr_cr, 50))   if len(arr_cr) else 0.0,
            "q3":   float(np.percentile(arr_cr, 75))   if len(arr_cr) else 0.0,
            "q4_max": float(np.percentile(arr_cr, 100)) if len(arr_cr) else 0.0,
            "sum":  float(sum(contrast_ratios))        if contrast_ratios else 0.0,
        }
        return summary, info_list
