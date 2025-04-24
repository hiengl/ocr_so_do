import json

import numpy as np
import yaml
from pathlib import Path

from skimage import io, color
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
from skimage.transform import rotate

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_json(data: dict, output_path: str):
    """Save data to JSON file."""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sort_text_by_position(text_list, line_threshold=15, word_threshold=20):
    """
    Sắp xếp danh sách văn bản theo vị trí (x, y) của từng từ trong ảnh.
    :param text_list: Danh sách các tuple chứa vị trí và văn bản
    :param line_threshold: Ngưỡng để xác định các dòng văn bản gần nhau
    :param word_threshold: Ngưỡng để xác định các từ gần nhau
    :return: Danh sách văn bản đã được sắp xếp
    """

    if not text_list:
        # logger.warning("Empty text list provided to sort_text_by_position")
        return []
    # Sắp xếp kết quả theo vị trí dọc (y-coordinate)
    text_list.sort(key=lambda x: int(x[0][0][1]), reverse=False)

    # Nhóm các dòng văn bản thành các cụm từ có nghĩa
    grouped_text = []
    current_group = []
    current_center_y = sum([text_list[0][0][i][1] for i in range(4)]) / 4

    try:
        for line in text_list:
            center_y = sum([line[0][i][1] for i in range(4)]) / 4
            if abs(center_y - current_center_y) < line_threshold:
                current_group.append(line)
            else:
                # Sắp xếp các từ trong dòng theo vị trí ngang (x-coordinate)
                current_group.sort(key=lambda x: int(x[0][0][0]))
                if len(current_group) > 1:
                    distance_bef_word = [
                        int(current_group[idx + 1][0][0][0] - current_group[idx][0][1][0] > word_threshold) for idx in
                        range(len(current_group) - 1)]
                    if sum(distance_bef_word) < 1:
                        grouped_text.append(" ".join([word[1] for word in current_group]))
                    else:
                        start_idx = 0
                        # Nếu có khoảng cách lớn giữa các từ, tách thành nhiều dòng
                        for idx in range(len(current_group) - 1):
                            if distance_bef_word[idx] == 1:
                                grouped_text.append(" ".join([word[1] for word in current_group[start_idx:idx + 1]]))
                                start_idx = idx + 1
                        grouped_text.append(" ".join([word[1] for word in current_group[start_idx:]]))
                else:
                    grouped_text.append(" ".join([word[1] for word in current_group]))
                current_group = [line]
            current_center_y = center_y

        if current_group:
            current_group.sort(key=lambda x: int(x[0][0][0]))
            grouped_text.append(" ".join([word[1] for word in current_group]))

    except Exception as e:
        # logger.error("Failed to sort text by position: %s", str(e), exc_info=True)
        return []

    return grouped_text


def parse_json_from_code_block(s: str) -> dict:
    """
    Trích xuất và parse JSON từ một chuỗi có chứa block kiểu ```json ... ```
    hoặc chỉ ``` ... ```.
    """
    # Loại bỏ dấu ` ở đầu và cuối
    cleaned = s.strip('`').strip()

    # Nếu có prefix 'json', loại bỏ
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

    # Parse JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi parse JSON: {e}") from e


def rotate_image(image=None, angle=None):
    """
    Xoay ảnh theo góc đã cho.
    :param image: Ảnh đầu vào (PIL Image)
    :param angle: Góc xoay (độ)
    :return: Ảnh đã xoay
    """


    image = io.imread('/Users/admin/Documents/OCR/ocr_so_do/data/temp/page_2.png')
    gray = color.rgb2gray(image)
    edges = canny(gray)

    h, theta, d = hough_line(edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    a = mode(angles)
    angle = np.rad2deg(mode(angles)[0][0])

    rotated = rotate(image, angle, resize=True)
    return image.rotate(angle, expand=True)

if __name__ == "__main__":
    rotate_image()