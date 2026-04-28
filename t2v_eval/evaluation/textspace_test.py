"""
Accessibility metric: word-spacing compliance (WCAG 2.x).
"""

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image


class TextSpaceTest:
    """
    Test whether word spacing in an image meets the WCAG 2.x criterion.

    Reference
    ---------
    https://www.w3.org/WAI/WCAG22/quickref/?showtechniques=141%2C145#text-spacing

    Parameters
    ----------
    image_path             : path to the image to analyse
    word_spacing_criteria  : minimum required gap as a fraction of font height
                             (default 0.16)
    """

    def __init__(self, image_path: str = "", word_spacing_criteria: float = 0.16):
        self.image_path            = image_path
        self.word_spacing_criteria = word_spacing_criteria

    def run_test(self, verbose: bool = False) -> tuple[dict, list]:
        """
        Run the word-spacing test.

        Returns
        -------
        (summary_results, info_list)
            *summary_results* contains ``"score"`` (fraction of compliant word
            pairs) and detailed statistics.
            *info_list* contains per-pair dicts.
        """
        image = Image.open(self.image_path)
        data  = pd.DataFrame(
            pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        )
        image.close()

        data = data[data["text"].str.strip() != ""]

        numeric_cols = ["left", "top", "width", "height", "block_num", "word_num"]
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

        unsuccessful = 0
        info_list    = []
        labels       = []
        distances    = []

        for _, group in data.groupby("block_num"):
            group = group.sort_values("word_num").reset_index(drop=True)
            for i in range(len(group) - 1):
                w1, w2 = group.iloc[i], group.iloc[i + 1]
                gap         = w2["left"] - (w1["left"] + w1["width"])
                font_size   = w1["height"]
                threshold   = self.word_spacing_criteria * font_size

                label = 0
                if gap < threshold:
                    unsuccessful += 1
                    label = 1
                    if verbose:
                        print(f"  '{w1['text']}' → '{w2['text']}': "
                              f"gap={gap:.1f} < threshold={threshold:.1f}")

                info_list.append({
                    "distance":  float(gap),
                    "font_size": float(font_size),
                    "space":     float(threshold),
                    "label":     int(label),
                })
                labels.append(label)
                distances.append(gap - threshold)

        n_blocks = len(data["block_num"].unique())
        try:
            score = 1.0 - unsuccessful / n_blocks
        except ZeroDivisionError:
            score = 1.0

        arr_dist   = np.array(distances)
        arr_labels = np.array(labels)
        summary = {
            "score":       score,
            "success":     int(np.count_nonzero(arr_labels == 0)),
            "failed":      int(np.count_nonzero(arr_labels == 1)),
            "text_number": n_blocks,
            "max":   float(np.max(arr_dist))             if len(arr_dist) else 0.0,
            "min":   float(np.min(arr_dist))             if len(arr_dist) else 0.0,
            "avg":   float(np.mean(arr_dist))            if len(arr_dist) else 0.0,
            "q1":    float(np.percentile(arr_dist, 25))  if len(arr_dist) else 0.0,
            "q2":    float(np.percentile(arr_dist, 50))  if len(arr_dist) else 0.0,
            "q3":    float(np.percentile(arr_dist, 75))  if len(arr_dist) else 0.0,
            "q4_max": float(np.percentile(arr_dist, 100)) if len(arr_dist) else 0.0,
            "sum":   float(np.sum(arr_dist))             if len(arr_dist) else 0.0,
        }
        return summary, info_list
