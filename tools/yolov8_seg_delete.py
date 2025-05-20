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
from ultralytics import YOLO
from insightface.app import FaceAnalysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from few_shot.few_shot_gender import predict_gender_fewshot
from yolov7.utils.plots import plot_one_box

prototypes = np.load("Pretrained/proto_new.npy", allow_pickle=True).item()

# Ó³Éä±í
gender_map = ["M", "F"]
age_map = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
race_map = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

fairface_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def detect(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # Load YOLOv8-seg model
    model = YOLO(opt.weights)

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
    img_files = list(Path(opt.source).glob("*.jpg")) + list(Path(opt.source).glob("*.png"))+ list(Path(opt.source).glob("*.jpeg"))

    for img_path in img_files:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            continue
        r = model(str(img_path), classes=[0])[0] # only person
        boxes = r.boxes.xyxy.cpu().numpy()
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []


        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2 = map(int, box)
            mask_resized = cv2.resize(mask, (im0.shape[1], im0.shape[0]))
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

            masked_img = cv2.bitwise_and(im0, im0, mask=mask_bin)
            person_crop = masked_img[y1:y2, x1:x2]

            face_results = face_app.get(person_crop)
            if face_results:
                if len(face_results) > 1:
                    print("!!!!!! more than one face lahhhh")
                face = face_results[0]
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
                        fewshot_result = predict_gender_fewshot(face.embedding, prototypes, id=idx + 1)
                        if fewshot_result == 'neutral_female':
                            gender_str = 'F-n'

                    age_str = age_map[age] if 0 <= age < len(age_map) else 'U'
                    race_str = race_map[race] if 0 <= race < len(race_map) else 'U'

                    label = f"{idx + 1}:{gender_str}"
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[idx % len(colors)], line_thickness=2)
                    colored_mask = np.zeros_like(im0)
                    color = (0, 255, 0)
                    for c in range(3):
                        colored_mask[:, :, c] = mask_bin * color[c] // 255
                    im0 = cv2.addWeighted(im0, 1.0, colored_mask, 0.5, 0)
                    results.append([img_path.name, idx + 1, gender_str, age_str, race_str])
                except Exception as e:
                    print(f"[!] Face crop error: {e}")

        save_path = str(save_dir / img_path.name)
        cv2.imwrite(save_path, im0)
        print(f"Saved: {save_path}")

    csv_path = str(save_dir / 'image_results.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'PersonID', 'Gender', 'Age', 'Race'])
        writer.writerows(results)
    print(f"CSV saved: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s-seg.pt', help='YOLOv8-seg model path')
    parser.add_argument('--source', type=str, default='inference/images', help='Image folder to process')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on')
    parser.add_argument('--project', default='runs/detect', help='Project folder')
    parser.add_argument('--name', default='exp', help='Experiment name')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)
