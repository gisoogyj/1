# -*- coding: utf-8 -*-
import os
import sys
import cv2
import csv
import torch
import argparse
import torchvision
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, increment_path, non_max_suppression, scale_coords, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized
from insightface.app import FaceAnalysis

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'few_shot')))
from few_shot.few_shot_gender import predict_gender_fewshot
prototypes = np.load("Pretrained/proto.npy", allow_pickle=True).item()

# Ó³Éä±í
gender_map = ["M", "F"]
age_map = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
race_map = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

# FairFace transform
fairface_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def select_top_face(faces, person_box):
    if len(faces) == 1:
        return faces[0]
    
    x1, y1, x2, y2 = person_box
    person_center_x = (x1 + x2) / 2

    def face_center(face):
        fx1, fy1, fx2, fy2 = face.bbox
        face_center_x = (fx1 + fx2) / 2
        return abs(face_center_x - person_center_x)

    return min(faces, key=face_center)


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

    # Load FairFace model
    fairface_model = torchvision.models.resnet34(pretrained=True)
    fairface_model.fc = nn.Linear(fairface_model.fc.in_features, 18)
    fairface_model.load_state_dict(torch.load('Pretrained/res34_fair_align_multi_7_20190809.pt', map_location=device))
    fairface_model = fairface_model.to(device).eval()

    # Load InsightFace
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0)

    results = []
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Run detection
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

                face_results = face_app.get(person_crop)
                if len(face_results) > 0:
                    face = select_top_face(face_results, [x1, y1, x2, y2])
                    try:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        face_crop = cv2.resize(face_crop, (224, 224))
                        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_tensor = fairface_transform(face_crop).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = fairface_model(face_tensor).cpu().numpy().squeeze()

                        race = int(np.argmax(output[:7]))
                        gender = int(np.argmax(output[7:9]))
                        age = int(np.argmax(output[9:18]))

                        gender_str = gender_map[gender] if gender in [0, 1] else 'U'
                        if gender_str == 'M':
                            fewshot_result = predict_gender_fewshot(face.embedding, prototypes)
                            if fewshot_result == 'neutral_female':
                                gender_str = 'F-n'

                        age_str = age_map[age] if 0 <= age < len(age_map) else 'U'
                        race_str = race_map[race] if 0 <= race < len(race_map) else 'U'
                        
                        label = f"{idx + 1}:{gender_str}"
                        plot_one_box(xyxy, im0, label=label, color=colors[idx % len(colors)], line_thickness=2)
                        results.append([p.name, idx + 1, gender_str, age_str, race_str])
                    except Exception as e:
                        print(f'[!] Face crop error: {e}')

        # Save image
        save_path = str(save_dir / p.name)
        cv2.imwrite(save_path, im0)
        print(f"Saved: {save_path}")

    # Save CSV
    csv_path = str(save_dir / 'image_results.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'PersonID', 'Gender', 'Age', 'Race'])
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
