# ───────────────────────────────────────────────────────────────
# 셀 하나에 이걸로 끝! “타깃 제외 모자이크” 실시간 파이프라인 (디버깅 포함)

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 0) 저장해둔 센트로이드 로드
centroid = np.load(r'C:\Users\nsjic\OneDrive\바탕 화면\centroid.npy')  # 필요시 절대경로

# 1) 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo   = YOLO(r'C:\Users\nsjic\OneDrive\바탕 화면\yolov8n.pt')  # 필요시 절대경로
mtcnn  = MTCNN(
    image_size=224, margin=40,
    thresholds=[0.5,0.6,0.7], min_face_size=20,
    keep_all=False, device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 2) 모자이크 함수
def mosaic(img, x1, y1, x2, y2, scale=0.05):
    sub = img[y1:y2, x1:x2]
    if sub.size == 0: return
    h, w = sub.shape[:2]
    small = cv2.resize(sub, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# 3) 웹캠 열기 (Windows에서는 CAP_DSHOW 권장)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 터미널이나 VS Code에서 실행해 보세요.")
print("▶ 실시간 모자이크 시작 (‘p’ 키로 종료)")

# 4) 임계값 설정 (코사인 유사도 기준)
THRESHOLD = 0.30

# 5) 실시간 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5-1) YOLO로 사람 검출
    result = yolo(frame, conf=0.5, iou=0.45)[0]
    persons = [b for b in result.boxes if int(b.cls) == 0]

    for b in persons:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # 5-2) 얼굴 정렬 & 임베딩 비교
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            mosaic(frame, x1, y1, x2, y2)
            continue

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        face = mtcnn(pil)
        if face is None:
            mosaic(frame, x1, y1, x2, y2)
            continue

        with torch.no_grad():
            emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()

        # 5-3) 코사인 유사도 계산 & 디버깅 출력
        cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))
        print(f"[Debug] cos_sim = {cos_sim:.3f}")

        # 5-4) 기준 이하일 때 모자이크, 이상일 때 박스
        if cos_sim < THRESHOLD:
            mosaic(frame, x1, y1, x2, y2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 5-5) 결과 표시
    cv2.imshow('Mask Except Target', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()