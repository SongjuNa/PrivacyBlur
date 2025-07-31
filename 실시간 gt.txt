import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=224, margin=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 등록자 평균 임베딩 불러오기
centroid = np.load(r'C:\Users\ed007\OneDrive\바탕 화면\centroid.npy')
THRESHOLD = 0.4  # 등록자 판별 기준

# 영상 경로 (바꿔주세요)
video_path = r'C:\Users\ed007\OneDrive\바탕 화면\your_video.mp4'

# 바탕화면 경로에 저장
desktop = os.path.join(os.path.expanduser("~"), 'OneDrive', '바탕 화면')
save_path = os.path.join(desktop, 'gt_labels.npy')

# Ground truth 생성
cap = cv2.VideoCapture(video_path)
gt_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)

    if face is None:
        gt_labels.append(0)
        continue

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()

    sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))
    gt_labels.append(int(sim >= THRESHOLD))

cap.release()

# 저장
np.save(save_path, np.array(gt_labels))
print(f"✅ Ground truth 저장 완료 → {save_path}")
