import math
import argparse
import numpy as np
import cv2
from paddleocr import PaddleOCR

from cv2.typing import MatLike
from typing import Tuple


def calculate_area(box: list) -> float:
    """计算文本框面积"""
    box = np.array(box)
    # 提取 x 和 y 坐标
    x = box[:, 0]  # x坐标
    y = box[:, 1]  # y坐标
    # 计算面积
    area = 0.5 * abs(np.sum(x * np.roll(y, 1) - y * np.roll(x, 1)))
    return area


def rotate_and_resize_image(image: MatLike, angle: float) -> MatLike:
    """旋转图像并调整尺寸"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像尺寸
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # 调整旋转矩阵的平移部分
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 旋转图像并调整尺寸
    rotated_image = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated_image


def rotate_90_image(image: MatLike) -> MatLike:
    """将图片旋转90度"""
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def visualize_boxes(image: MatLike, boxes: list, texts: list) -> MatLike:
    """Draw detected text boxes on the image."""
    # print(texts, "texts")
    image_with_boxes = image.copy()

    for box, text in zip(boxes, texts):
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        # 绘制文本框
        cv2.polylines(image_with_boxes, [box], True, color=(0, 255, 0), thickness=2)

        # 获取文本框中心点作为面积文本的位置
        center = np.mean(box, axis=0).astype(int)
        cv2.putText(
            image_with_boxes,
            str(text),
            tuple(center[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return image_with_boxes


def detect_box_orientation(image: MatLike) -> Tuple[int, bool]:
    """Detect the rotation angle needed to make box horizontal using PaddleOCR."""
    result = paddleocr.ocr(image, rec=False, det=True, cls=False)
    if not result or not result[0]:
        return 0, False
    boxes = result[0]

    horizontal_count = 0
    vertical_count = 0
    angle_list = []

    top_5_boxes = sorted(boxes, key=calculate_area, reverse=True)[:5]

    for box in top_5_boxes:
        # 文本框水平垂直分类
        width = max(abs(box[1][0] - box[0][0]), abs(box[2][0] - box[3][0]))
        height = max(abs(box[3][1] - box[0][1]), abs(box[2][1] - box[1][1]))

        if width > height * 1.5:
            horizontal_count += 1
        elif height > width * 1.5:
            vertical_count += 1

        # 计算矫正到水平的角度
        x1, y1 = box[0]  # Left-top
        x2, y2 = box[1]  # Right-top

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle_list.append(int(angle))

    # det_box_ret = visualize_boxes(image, top_5_boxes, angle_list)
    # cv2.imwrite("det_box_ret.png", det_box_ret)

    print(f"Box angle: {angle_list}")

    # Normalize angle to (-90, 90)
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    if len(angle_list) > 2:
        angle_list.remove(max(angle_list))
        angle_list.remove(min(angle_list))
    return int(np.median(angle_list)), horizontal_count < vertical_count


def detect_text_orientation(image: MatLike) -> int:
    """Detect the rotation angle needed to make text horizontal using PaddleOCR."""
    result = paddleocr.ocr(image, rec=False, det=True, cls=False)
    if not result or not result[0]:
        return 0
    boxes = result[0]

    top_10_boxes = sorted(boxes, key=calculate_area, reverse=True)[:10]
    angle_list = []

    for box in top_10_boxes:
        points = np.array(box, dtype=np.int32)
        # 裁剪图像
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]
        # 识别文本角度
        angle_result = paddleocr.ocr(cropped_image, cls=True, det=False, rec=False)
        angle = angle_result[0][0][0]
        angle_list.append(int(angle))

    # det_text_ret = visualize_boxes(image, top_10_boxes, angle_list)
    # cv2.imwrite("det_text_ret.png", det_text_ret)

    most_angle = np.argmax(np.bincount(np.array(angle_list)))

    print(f"Text angle: {angle_list}")
    return most_angle


def main():
    """Process image to make text horizontal."""
    parser = argparse.ArgumentParser(description="图像文本水平矫正旋转工具")
    parser.add_argument("input_image_path", help="输入图像路径")
    parser.add_argument(
        "output_image_path",
        nargs="?",
        default="./output.png",
        help="输出图像路径，默认为 output.png",
    )
    args = parser.parse_args()

    global paddleocr
    paddleocr = PaddleOCR(use_angle_cls=True, lang="ch")

    image = cv2.imread(args.input_image_path)

    angle, flag = detect_box_orientation(image)
    print(f"Detected Box angle: {angle}, Text is vertical: {flag}")

    if angle != 0:
        image = rotate_and_resize_image(image, angle)

    if flag is True:
        image = rotate_90_image(image)

    angle = detect_text_orientation(image)
    print(f"Detected Text angle: {angle}")

    if angle == 180:
        image = rotate_and_resize_image(image, angle)
    print(args.output_image_path)
    
    cv2.imwrite(args.output_image_path, image)


if __name__ == "__main__":
    main()
