# -*- coding: utf-8 -*-
import os

# �������ͼƬ�ļ���·��
folder_path = r"E:/学习/flow/测试/TEST/images"

# ֧�ֵ�ͼƬ��չ���������ִ�Сд��
valid_exts = ['.jpg', '.jpeg', '.png']

# ��ȡ����ͼƬ�ļ�������
image_files = sorted([
    f for f in os.listdir(folder_path)
    if os.path.splitext(f)[1].lower() in valid_exts
])

# ����������Ϊ 1.jpg, 2.jpg, ...
for i, filename in enumerate(image_files, 1):
    ext = '.jpg'  # ����ͳһת�� .jpg ��չ��
    new_name = f"{i}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"{filename} -> {new_name}")
