from t2v_eval.evaluation.contrast_test import ContrastTest
from t2v_eval.evaluation.textspace_test import TextSpaceTest
import time

def calculate_accessibility_score(img_path, verbose=False):
    
    start_time = time.time()
    contrast_test = ContrastTest(image_path=img_path)
    contrast_score, _ = contrast_test.run_test(use_clusters='True', use_rgb='False', opt='2')
    if verbose:
        print(img_path)
        print(f"Contrast test completed in {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    space_test = TextSpaceTest(image_path=img_path)
    spacing_score, _ = space_test.run_test()
    if verbose:
        print(f"Text space test completed in {time.time() - start_time:.2f} seconds.")
    
    return {
        "contrast_score": contrast_score["score"],
        "spacing_score": spacing_score["score"]
    }
