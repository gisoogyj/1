# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ��ʼ�� InsightFace
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

def extract_embeddings_from_folder(folder_path):
    embeddings = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            faces = face_app.get(img)
            if faces:
                embeddings.append(faces[0].embedding)
            else:
                print(f"No face detected in {file}")
    return np.array(embeddings)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ����Ů������·��
neutral_female_folder = "TEST/Few/women/"
male_folder = "TEST/Few/men/"

# ��ȡ������ƽ�� embedding��ԭ�ͣ�
neutral_female_embeddings = extract_embeddings_from_folder(neutral_female_folder)
male_embeddings = extract_embeddings_from_folder(male_folder)

prototypes = {
    "neutral_female": np.mean(neutral_female_embeddings, axis=0),
    "male": np.mean(male_embeddings, axis=0)
}

# ����ԭ��Ϊ .npy �ļ�
np.save("Pretrained/proto_new.npy", prototypes)
print("Prototypes saved to proto.npy")
