import cv2
import numpy as np
import torch
from PIL import Image
import re
import easyocr
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1

# ─────────────────────────────────────
# 설정
THRESHOLD = 0.30
centroid = np.load(r'C:\Users\nsjic\OneDrive\바탕 화면\centroid.npy')

# ─────────────────────────────────────
# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO(r'C:\Users\nsjic\OneDrive\바탕 화면\yolov8n-seg.pt')  # segmentation model
mtcnn = MTCNN(image_size=224, margin=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
ocr = easyocr.Reader(['ko', 'en'])

# ─────────────────────────────────────
# 유틸리티 함수
def mosaic_mask(img, mask, scale=0.05):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3d = (mask > 0.5).astype(np.uint8)[:, :, None]
    img[:] = img * (1 - mask_3d) + mosaic * mask_3d

def draw_mask_contour(frame, mask, color=(255, 0, 0)):
    mask_bin = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)

def blur(img, x1, y1, x2, y2):
    sub = img[y1:y2, x1:x2]
    if sub.size == 0: return
    blurred = cv2.GaussianBlur(sub, (51, 51), 0)
    img[y1:y2, x1:x2] = blurred

def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}', r'01[0-9]-\d{3,4}-\d{4}',
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        r'\d{1,4}동|\d{1,4}호', r'[\uac00-\ud7a3]+[시군구동읍면로길]',
        r'\d{9,14}', r'(대학교|중학교|고등학교|회사|직장|소속)',
        r'(이름|성명)[:：]?\s?[가-힣]{2,4}'
    ]
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# ─────────────────────────────────────
# 실시간 처리
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

print("▶ 실시간 모자이크 시작 ('q' 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    results = yolo(frame, conf=0.5, iou=0.45)[0]

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()

        for mask, cls, box in zip(masks, classes, boxes):
            if cls != 0:
                continue

            mask_bin = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            draw_mask_contour(frame, mask, color=(255, 0, 0))

            # YOLO bbox 기반 crop (신뢰도 향상)
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            try:
                face = mtcnn(pil)
                if face is None:
                    raise ValueError
            except:
                mosaic_mask(frame, mask)
                continue

            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
            cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))
            print(f"[Debug] 얼굴 유사도: {cos_sim:.3f}")

            if cos_sim < THRESHOLD:
                mosaic_mask(frame, mask)

    # OCR 탐지 → 개인정보 블러
    texts = ocr.readtext(orig)
    for (bbox, text, conf) in texts:
        if is_sensitive_text(text):
            pts = np.array(bbox, dtype=np.int32)
            x1 = int(min(pt[0] for pt in pts))
            y1 = int(min(pt[1] for pt in pts))
            x2 = int(max(pt[0] for pt in pts))
            y2 = int(max(pt[1] for pt in pts))
            blur(frame, x1, y1, x2, y2)

    cv2.imshow('Privacy Protected Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
