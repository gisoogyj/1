# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import cv2
import csv
import torch
import argparse
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, increment_path, non_max_suppression, scale_coords, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device

from deepface import DeepFace

def detect(opt):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load YOLO model
    sys.path.append('./yolov7')
    model = attempt_load(opt.weights, map_location=device)
    if half:
        model.half()
    model.eval()

    # Load dataset
    dataset = LoadImages(opt.source, img_size=opt.img_size, stride=int(model.stride.max()))

    results = []
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0], agnostic=opt.agnostic_nms)[0]
        im0 = im0s.copy()
        p = Path(path)

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            for idx, (*xyxy, conf, cls) in enumerate(pred):
                x1, y1, x2, y2 = map(int, xyxy)
                person_crop = im0[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                try:
                    # ??? DeepFace ????????
                    analysis = DeepFace.analyze(person_crop, actions=['age', 'gender'], enforce_detection=False)

                    # gender = analysis[0]['gender']
                    gender = analysis[0]["dominant_gender"]
                    age = int(analysis[0]['age'])  # ????????????

                    gender_str = 'M' if gender.lower().startswith('m') else 'F'

                    label = f"ID-{idx + 1}: {gender_str} Age:{age}"
                    plot_one_box(xyxy, im0, label=label, color=colors[idx % len(colors)], line_thickness=2)
                    results.append([p.name, idx + 1, gender_str, age])

                except Exception as e:
                    print(f"[!] DeepFace error on ID-{idx + 1}: {e}")

        save_path = str(save_dir / p.name)
        cv2.imwrite(save_path, im0)
        print(f"Saved: {save_path}")

    csv_path = str(save_dir / 'image_results.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'PersonID', 'Gender', 'Age'])
        writer.writerows(results)
    print(f"CSV saved: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='Pretrained/yolov7.pt', help='YOLOv7 model path')
    parser.add_argument('--source', type=str, default='inference/images', help='Folder of images to process')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='CUDA device or "cpu"')
    parser.add_argument('--project', default='runs/detect', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='Augmented inference')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)
