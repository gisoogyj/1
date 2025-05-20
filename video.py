import cv2
import os

video_path = 'E:/shoopingmall.mp4'
output_folder = 'E:/outtttput'

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"第 {frame_idx} 帧读取失败")
        break

    if frame is None:
        print(f"第 {frame_idx} 帧是 None，不能保存")
    else:
        frame_filename = os.path.join(output_folder, f'frame_{frame_idx:04d}.jpg')
        success = cv2.imwrite(frame_filename, frame)
        if not success:
            print(f"第 {frame_idx} 帧保存失败！")
        else:
            print(f"保存成功: {frame_filename}")

    frame_idx += 1

cap.release()
print(f"完成！总共读取了 {frame_idx} 帧。")
