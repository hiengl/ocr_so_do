import os
import sys
from typing import List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch

from .models.text_detection.craft_process import CraftDetector
from .models.text_recognition.vietocr_process import Predictor
from .utils import sort_text_by_position


# Add the src directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


class OCRProcessor:
    def __init__(self, config):
        self.device = self._get_device(config["device"])
        self.config = config
        self.craft_detector = CraftDetector(config=config['craft'], device=self.device)
        self.vietocr_predictor = Predictor(config=config['vietocr'], device=self.device)
        self.batch_size = config["batch_size"]
        self.num_threads = config["num_threads"]
        self.temp_dir = config["temp_dir"]
        os.makedirs(self.temp_dir, exist_ok=True)

    def _get_device(self, device_str: str) -> torch.device:
        """Determine the device to use based on config and availability."""
        if "cuda" in device_str and torch.cuda.is_available():
            return torch.device(device_str)
        elif device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def process_image(self, image_path: str) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Process a single image: detect and recognize text."""
        image = self.craft_detector.load_image(image_path)
        bbox_detection, _ = self.craft_detector.detect_text(image)
        cropped_images = [self.craft_detector.crop_image(image, coord) for coord in bbox_detection]
        texts, recogn_probs = self.vietocr_predictor.predict_batch(cropped_images, return_prob=True)
        # sort the texts by their bounding box positions folling line
        sorted_text = sort_text_by_position([[bbox, texts, prob] for bbox, texts, prob in zip(bbox_detection, texts, recogn_probs)])

        return sorted_text

    def process_images(self, image_paths: List[str]) -> List[List[Tuple[str, Tuple[int, int, int, int]]]]:
        """Process multiple images with multi-threading."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(self.process_image, image_paths))
        return results


if __name__ == "__main__":
    pass