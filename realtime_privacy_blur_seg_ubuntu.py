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

import os, csv
os.makedirs("frames", exist_ok=True)
frame_ids = []
gt_labels = []
gt_texts = []
frame_id = 0

#---------
#camera index definition
def find_camera_index():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    raise RuntimeError("not find camera")


camera_index = find_camera_index()

# ─────────────────────────────────────
# 설정
THRESHOLD = 0.40
centroid = np.load('/home/server4/Downloads/centroid_jw.npy')

# ─────────────────────────────────────
# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO('/home/server4/Downloads/yolov8n-seg.pt')  # segmentation model
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

def blur_polygon(image, polygon, ksize=(51, 51)):
    """
    polygon: easyocr에서 반환된 4개의 꼭짓점 좌표 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)

    blurred = cv2.GaussianBlur(image, ksize, 0)
    mask_3ch = cv2.merge([mask] * 3)

    result = np.where(mask_3ch == 255, blurred, image)
    return result

def is_sensitive_text(text):
    patterns = [
    #user info
    r'\d{6}-\d{7}',                         #  주민등록번호 (예: 900101-1234567)
    r'01[0-9]-\d{3,4}-\d{4}',              #  휴대폰 번호 (예: 010-1234-5678)
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  #  이메일 주소
    #location info
    r'\d{1,4}동|\d{1,4}호',                #  아파트 호수 (예: 302동, 1502호)
    r'[\uac00-\ud7a3]+[시군구동읍면로길]', #  도로명 주소 일부 (예: 강남구, 서초동, 종로길)
    r'[\uac00-\ud7a3]+[시군구동읍면로길]',         # 지역 주소 단어
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|맨션|오피스텔|연립)',  # 건물 유형
        r'\d{1,4}-\d{1,4}',                            # 지번 주소

    
    r'\d{9,14}',                           #  사업자 등록번호 / 계좌번호 / 전화번호 등 (9~14자리 숫자열)
    r'(대학교|중학교|고등학교|회사|직장|소속)', #  소속기관 (학교, 회사 등)
    
    r'(이름|성명)[:：]?\s?[가-힣]{2,4}' ,    #  이름 표기 (예: 이름: 김민수, 성명：이영희)

    r'\d{6}-\d{7}',                                # 주민등록번호
        r'01[0-9]-\d{3,4}-\d{4}',                      # 휴대폰 번호
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  # 이메일 주소

        # 계좌번호 패턴
        r'\d{2,4}-\d{2,4}-\d{4,7}',                    # 일반 은행형 계좌번호
        r'\d{9,14}'                                # 숫자만 있는 계좌 
 ]
     
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# ─────────────────────────────────────
# 실시간 처리
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

print("▶ 실시간 모자이크 시작 ('q' 키로 종료)")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - start_time > 30:
        print("30초 경과. 종료합니다.")
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
            frame = blur_polygon(frame, bbox)

    cv2.imshow('Privacy Protected Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#---------------------------------------
#[1] 얼굴 탐지 정확도 평가        
def evaluate_face_detection(gt_labels, pred_labels):
    tp = np.sum((gt_labels == 1) & (pred_labels == 1))
    tn = np.sum((gt_labels == 0) & (pred_labels == 0))
    fp = np.sum((gt_labels == 0) & (pred_labels == 1))
    fn = np.sum((gt_labels == 1) & (pred_labels == 0))
    acc = (tp + tn) / len(gt_labels)
    print(f"\n[1. 얼굴 탐지 정확도 평가]")
    print(f" - 정탐(TP): {tp}, 오탐(FP): {fp}, 미탐(FN): {fn}, 정음(TN): {tn}")
    print(f" - Accuracy: {acc:.3f}")

# ─────────────────────────────────────
# [2] ArcFace 유사도 감소율
def arcface_drop(original_scores, blurred_scores):
    original_scores = np.array(original_scores)
    blurred_scores = np.array(blurred_scores)
    drop_rates = original_scores - blurred_scores
    avg_drop = np.mean(drop_rates)
    print(f"\n[2. ArcFace 유사도 감소율]")
    print(f" - 개별 감소율: {drop_rates}")
    print(f" - 평균 감소율: {avg_drop:.3f}")

# ─────────────────────────────────────
# [3] 민감 텍스트 탐지 정확도
def arcface_drop(original_scores, blurred_scores):
    original_scores = np.array(original_scores)
    blurred_scores = np.array(blurred_scores)
    drop_rates = original_scores - blurred_scores
    avg_drop = np.mean(drop_rates)
    print(f"\n[2. ArcFace 유사도 감소율]")
    print(f" - 개별 감소율: {drop_rates}")
    print(f" - 평균 감소율: {avg_drop:.3f}")

# ─────────────────────────────────────
# [4] 실시간 처리 속도 측정
def measure_speed(func, frame, repeat=30):
    times = []
    for _ in range(repeat):
        start = time.time()
        func(frame)
        end = time.time()
        times.append(end - start)
    avg_latency = np.mean(times)
    fps = 1 / avg_latency
    print(f"\n[4. 실시간 처리 속도]")
    print(f" - 평균 Latency: {avg_latency * 1000:.2f} ms")
    print(f" - 평균 FPS:     {fps:.2f}")

        
cap.release()
cv2.destroyAllWindows()