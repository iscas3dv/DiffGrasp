# import imageio

# with imageio.get_writer(uri='test.gif', mode='I', duration = 1000, loop=0) as writer:
#     for i in range(3):
#         writer.append_data(imageio.imread(f'C:\\Users\\zhang\\Desktop\\webpage\\handpose-virtualview\\resources\\{i+1}.jpeg'))

import imageio

input_video = r"C:\Users\zhang\Desktop\AAAI2025作图\supp\video_demo\pr\resource\aaai_render\1290\1290-0_gt_obj_only.mp4"
output_gif = r"C:\Users\zhang\Desktop\AAAI2025作图\supp\video_demo\pr\resource\aaai_render\1290\1290-0_GRAB_AAAI_final_all_refine.gif"


# reader = imageio.get_reader(input_video)
# frames = [frame for frame in reader]


# with imageio.get_writer(uri=output_gif, mode='I', duration = 100, loop=0) as writer:
#     for f in frames:
#         writer.append_data(f)

import cv2

# 读取视频
video_path = input_video
cap = cv2.VideoCapture(video_path)

frame_count = 0
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 保存为 PNG 格式，确保无损
    cv2.imwrite(f"frame_{frame_count}.png", frame)
    frames.append(frame)
    frame_count += 1

cap.release()

with imageio.get_writer(uri=output_gif, mode='I', duration = 100, loop=0) as writer:
    for i in range(30):
        writer.append_data(imageio.imread(f'C:\\Users\\zhang\\Desktop\\AAAI2025作图\\supp\\video_demo\\pr\\resource\\aaai_render\\1290\\frame_{i}.png'))

# with imageio.get_writer(uri=output_gif, mode='I', duration = 100, loop=0) as writer:
#     for f in frames:
#         writer.append_data(f)