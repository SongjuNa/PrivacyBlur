import cv2
import numpy as np
import torch
from PIL import Image
import re
import easyocr
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# ----------------------------
# 설정
THRESHOLD = 0.40
centroid = np.load(r'C:\Users\ed007\centroid.npy')

# 평가용 pred 저장 리스트
pred_labels = []

# ----------------------------
# 모델 초기화
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO(r'C:\Users\ed007\yolov8n-seg.pt')
mtcnn = MTCNN(image_size=224, margin=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
ocr = easyocr.Reader(['ko', 'en'])

# ----------------------------
# 유틸 함수들
def mosaic_mask(img, mask, scale=0.05):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3d = (mask > 0.5).astype(np.uint8)[:, :, None]
    img[:] = img * (1 - mask_3d) + mosaic * mask_3d

def blur_polygon(image, polygon, ksize=(51, 51)):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    blurred = cv2.GaussianBlur(image, ksize, 0)
    mask_3ch = cv2.merge([mask] * 3)
    return np.where(mask_3ch == 255, blurred, image)

def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}', r'01[0-9]-\d{3,4}-\d{4}',
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        r'\d{1,4}동|\d{1,4}호', r'[\uac00-\ud7a3]+[시군구동읍면로길]',
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|맨션|오피스텔|연립)',
        r'\d{1,4}-\d{1,4}', r'\d{2,4}-\d{2,4}-\d{4,7}', r'\d{9,14}',
        r'(대학교|중학교|고등학교|회사|직장|소속)',
        r'(이름|성명)[:：]?\s?[가-힣]{2,4}'
    ]
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# ----------------------------
# 실시간 루프
print("▶ 실시간 모자이크 시작 ('q' 키로 종료)")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    results = yolo(frame, conf=0.5, iou=0.45)[0]
    frame_pred_label = 0  # 기본: 등록자라고 가정

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()

        for mask, cls, box in zip(masks, classes, boxes):
            if cls != 0:
                continue
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
                frame_pred_label = 1
                continue

            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
            cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))

            if cos_sim < THRESHOLD:
                mosaic_mask(frame, mask)
                frame_pred_label = 1

    pred_labels.append(frame_pred_label)

    texts = ocr.readtext(orig)
    for (bbox, text, conf) in texts:
        if is_sensitive_text(text):
            frame = blur_polygon(frame, bbox)

    cv2.imshow('Privacy Protected Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 저장
save_path = "C:/Users/ed007/OneDrive/바탕 화면/pred_labels.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, np.array(pred_labels))
cap.release()
cv2.destroyAllWindows()

# ----------------------------
# 평가 함수

def evaluate_face_detection():
    gt_labels = np.load(r"C:/Users/ed007/OneDrive/바탕 화면/gt_labels.npy")
    pred_labels = np.load(r"C:/Users/ed007/OneDrive/바탕 화면/pred_labels.npy")    
    min_len = min(len(gt_labels), len(pred_labels))
    gt_labels = gt_labels[:min_len]
    pred_labels = pred_labels[:min_len]
    
    tp = np.sum((gt_labels == 1) & (pred_labels == 1))
    tn = np.sum((gt_labels == 0) & (pred_labels == 0))
    fp = np.sum((gt_labels == 0) & (pred_labels == 1))
    fn = np.sum((gt_labels == 1) & (pred_labels == 0))
    acc = (tp + tn) / len(gt_labels)
    
    print(f"\n[1. 얼굴 탐지 정확도 평가]")
    print(f" - 정탐(TP): {tp}, 오탐(FP): {fp}, 미탐(FN): {fn}, 정음(TN): {tn}")
    print(f" - Accuracy: {acc:.3f}")


# 평가 실행
evaluate_face_detection()
