# -*- coding: utf-8 -*-

import sys
import os
import cv2
import csv
import torch
import argparse
import time
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np
from numpy import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from yolov7.utils.plots import plot_one_box
from tracker.mc_bot_sort import BoTSORT
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.torch_utils import select_device
from few_shot.few_shot_gender import predict_gender_fewshot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gender_map = ["M", "F"]
age_map = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
race_map = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

# transform????
fairface_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def is_in_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]

def compute_containment(inner, outer):
    xi1, yi1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    xi2, yi2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter_area / inner_area if inner_area > 0 else 0

def save_statistics(save_dir, id_records, flow_records):
    save_dir = Path(save_dir)
    flow_res_dir = save_dir / 'statistics'
    flow_res_dir.mkdir(parents=True, exist_ok=True)

    flow_res_file = flow_res_dir / 'flow_count.txt'
    stay_res_file = flow_res_dir / 'stay_times.txt'
    long_stay_res_file = flow_res_dir / 'stay_long_analysis.txt'

    with open(flow_res_file, 'w') as f:
        f.write("Time Range, Number of People\n")
        f.writelines(flow_records)
    print(f"Flow count saved to {flow_res_file}")

    with open(stay_res_file, 'w') as f:
        f.write("ID, Entry Time (s), Exit Time (s), Stay Duration (s)\n")
        stay_long_ids = []
        for tid, (enter, leave) in id_records.items():
            stay_duration = leave - enter
            f.write(f"{tid},{enter:.2f},{leave:.2f},{stay_duration:.2f}\n")
            if stay_duration >= 2:
                stay_long_ids.append(tid)
    print(f"Stay duration records saved to {stay_res_file}")

    with open(long_stay_res_file, 'w') as f:
        f.write(f"Number of people stayed >=2s: {len(stay_long_ids)}\n")
        f.write(f"IDs: {stay_long_ids}\n")
    print(f"Long stay analysis saved to {long_stay_res_file}")

def detect(opt):
    # source, weights, project, name, device_str, nosave = opt.source, opt.weights, opt.project, opt.name, opt.device, opt.nosave
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    # yolo
    model = YOLO(opt.weights)

    # insightface
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0)

    # FairFace
    fairface_model = torchvision.models.resnet34(pretrained=True)
    fairface_model.fc = nn.Linear(fairface_model.fc.in_features, 18)
    fairface_model.load_state_dict(torch.load('Pretrained/res34_fair_align_multi_7_20190809.pt', map_location=device))
    fairface_model = fairface_model.to(device)
    fairface_model.eval()

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    prototypes = np.load("Pretrained/proto_new.npy", allow_pickle=True).item()

    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    roi = (658, 386, 1296, 986)

    id_face_buffer = defaultdict(list)       # ???id?????
    final_age_gender_race = dict()           # ??????
    buffer_max_len = 5                       # ??????????(???)???????????
    window_size = 5
    id_records = {}
    current_window_start = 0
    current_window_end = window_size
    flow_records = []

    t0 = time.time()

    cap = cv2.VideoCapture(opt.source)
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_path = str(save_dir / (Path(opt.source).stem + ".mp4"))
    vid_writer = None

    with tqdm(total=frame_count, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            im0 = frame.copy()
            current_time = frame_idx / fps
            frame_idx += 1

            # Detection
            r = model.predict(frame, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []
            cls_ids = r.boxes.cls.cpu().numpy()

            person_boxes = []
            person_masks = []
            person_confs = []
            for box, mask, conf, cls_id in zip(boxes, masks, confs, cls_ids):
                if int(cls_id) == 0:
                    person_boxes.append(box)
                    person_masks.append(mask)
                    person_confs.append(conf)

            # Remove duplicate boxes
            filtered_boxes, filtered_masks, filtered_confs = [], [], []
            for i in range(len(person_boxes)):
                keep = True
                for j in range(len(filtered_boxes)):
                    contain1 = compute_containment(person_boxes[i], filtered_boxes[j])
                    contain2 = compute_containment(filtered_boxes[j], person_boxes[i])
                    if max(contain1, contain2) > 0.85:
                        keep = False
                        break
                if keep:
                    filtered_boxes.append(person_boxes[i])
                    filtered_masks.append(person_masks[i])
                    filtered_confs.append(person_confs[i])

            detections = []
            for box, conf in zip(filtered_boxes, filtered_confs):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, conf, 0])
            detections = np.array(detections)

            # Tracking
            online_targets = tracker.update(detections, im0)

            for t in online_targets:
                tlbr = t.tlbr
                tid = t.track_id
                bbox = [int(x) for x in tlbr]

                # count
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                if roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]:
                    if tid not in id_records:
                        id_records[tid] = [current_time, current_time]
                    else:
                        id_records[tid][1] = current_time

                if tid not in final_age_gender_race:
                    x1, y1, x2, y2 = map(int, bbox)
                    if x2 <= x1 or y2 <= y1:
                        print(f"Invalid bbox: {(x1, y1, x2, y2)}")
                        continue
                    person_crop = im0[y1:y2, x1:x2]
                    if person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                        print(f"[Warning] Empty crop for tid {tid}, bbox: {(x1, y1, x2, y2)}")
                        continue               

                    face_results = face_app.get(person_crop)
                    if face_results:
                        if len(face_results) > 1:
                            print(f"[Warning] More than one face detected for ID {tid}")
                        face = face_results[0]
                        try:
                            fx1, fy1, fx2, fy2 = map(int, face.bbox)
                            face_crop = person_crop[fy1:fy2, fx1:fx2]
                            face_crop = cv2.resize(face_crop, (224, 224))
                            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_tensor = fairface_transform(face_crop).unsqueeze(0).to(device)

                            with torch.no_grad():
                                output = fairface_model(face_tensor).cpu().numpy().squeeze()
                            
                            race_idx = int(np.argmax(output[:7]))
                            gender_idx = int(np.argmax(output[7:9]))
                            age_idx = int(np.argmax(output[9:18]))

                            if gender_idx == 0: 
                                fewshot_result = predict_gender_fewshot(face.embedding, prototypes, id=tid)
                                if fewshot_result == 'neutral_female':
                                    gender_idx = 1 

                            id_face_buffer[tid].append((age_idx, gender_idx, race_idx))

                            if len(id_face_buffer[tid]) >= buffer_max_len:
                                ages, genders, races = zip(*id_face_buffer[tid])
                                final_age_idx = Counter(ages).most_common(1)[0][0]
                                final_gender_idx = Counter(genders).most_common(1)[0][0]
                                final_race_idx = Counter(races).most_common(1)[0][0]

                                final_age = age_map[final_age_idx] if 0 <= final_age_idx < len(age_map) else 'U'
                                final_gender = gender_map[final_gender_idx] if final_gender_idx in [0, 1] else 'U'
                                final_race = race_map[final_race_idx] if 0 <= final_race_idx < len(race_map) else 'U'

                                final_age_gender_race[tid] = (final_age, final_gender, final_race)

                        except Exception as e:
                            print(f"Face crop error for ID {tid}: {e}")
                    
                    else:
                        id_face_buffer[tid].append((-1, -1, -1))
                        if len(id_face_buffer[tid]) >= buffer_max_len:
                            final_age_gender_race[tid] = (-1, -1, -1)
           
            if current_time >= current_window_end:
                ids_in_window = [
                    tid for tid, (enter, leave) in id_records.items()
                    if leave >= current_window_start and enter <= current_window_end
                ]
                count = len(set(ids_in_window))
                print(f"{int(current_window_start)}s-{int(current_window_end)}s flow: {count}")
                flow_records.append(f"{int(current_window_start)}-{int(current_window_end)}s, count: {count}\n")

                current_window_start += window_size
                current_window_end += window_size

            for t in online_targets:
                tid = t.track_id
                tlbr = t.tlbr
                label = f"ID:{tid} " + " ".join(map(str, final_age_gender_race.get(tid, ("U", "U", "U"))))
                plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

            # Save output video
            if vid_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (im0.shape[1], im0.shape[0]))
            vid_writer.write(im0)

            pbar.update(1)

    cap.release()
    vid_writer.release()
    save_statistics(save_dir, id_records, flow_records)

    print(f"Finished, total time: {time.time() - t0:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='Pretrained/yolov8s-seg.pt', help='Path to YOLOv8-seg model')
    parser.add_argument('--source', type=str, default='inference/videos/test.mp4', help='Video file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run inference on, e.g. cuda:0 or cpu')
    parser.add_argument('--project', default='runs/detect', help='Directory to save results')
    parser.add_argument('--name', default='exp', help='Name of the current experiment')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument('--ablation', default=False, action='store_true', help='Ablation study mode (default False)')


    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        detect(opt)