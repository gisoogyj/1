# fewshot_gender.py

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predict_gender_fewshot(face_embedding, prototypes, threshold=0.08, id=None):
    sim_nf = cosine_similarity(face_embedding, prototypes["neutral_female"])
    sim_male = cosine_similarity(face_embedding, prototypes["male"])
    diff = sim_nf - sim_male

    # if id is not None and diff > threshold:
    #     print(f"[ID {id}] sim_neutral_female = {sim_nf:.3f}, sim_male = {sim_male:.3f}, diff = {diff:.3f}")

    if sim_nf > sim_male and diff > threshold and sim_male > 0.008:
        if id is not None:
            print(f"[ID {id}] sim_neutral_female = {sim_nf:.6f}, sim_male = {sim_male:.6f}, diff = {diff:.6f}")
            print(f"[ID {id}] changed to Female")
        return "neutral_female"
    else:
        return "male"
