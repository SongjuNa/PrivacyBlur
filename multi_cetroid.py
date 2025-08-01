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
centroids = np.load(r'C:/Users/ed007/OneDrive/바탕 화면/noise_centroid_multi.npy')  # shape: (N, 512)
pred_labels = []  # 평가용 pred 저장

# ----------------------------
# 폴더 생성
os.makedirs("C:/Users/ed007/OneDrive/바탕 화면/before_blur", exist_ok=True)
os.makedirs("C:/Users/ed007/OneDrive/바탕 화면/after_blur", exist_ok=True)

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
# 유틸 함수

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

def is_registered(embedding, centroids, threshold=THRESHOLD):
    sims = np.dot(centroids, embedding) / (np.linalg.norm(centroids, axis=1) * np.linalg.norm(embedding) + 1e-10)
    return np.max(sims) >= threshold

# ----------------------------
# 실시간 루프 시작
print("▶ 실시간 모자이크 시작 ('q' 키로 종료)")
total_frames = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    results = yolo(frame, conf=0.5, iou=0.45)[0]
    frame_pred_label = 0

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

            if not is_registered(emb, centroids, THRESHOLD):
                mosaic_mask(frame, mask)
                frame_pred_label = 1
            else:
                face_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                save_name = f"face_{int(time.time() * 1000)}.jpg"
                before_path = os.path.join("C:/Users/ed007/OneDrive/바탕 화면/before_blur", save_name)
                after_path = os.path.join("C:/Users/ed007/OneDrive/바탕 화면/after_blur", save_name)
                cv2.imwrite(before_path, face_bgr)
                frame_mosaic = frame.copy()
                mosaic_mask(frame_mosaic, mask)
                crop_after = frame_mosaic[y1:y2, x1:x2]
                cv2.imwrite(after_path, crop_after)

    pred_labels.append(frame_pred_label)

    texts = ocr.readtext(orig)
    for (bbox, text, conf) in texts:
        if is_sensitive_text(text):
            frame = blur_polygon(frame, bbox)

    total_frames += 1
    cv2.imshow('Privacy Protected Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 저장 및 FPS 계산
save_path = "C:/Users/ed007/OneDrive/바탕 화면/pred_labels.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, np.array(pred_labels))
cap.release()
cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time
avg_fps = total_frames / total_time if total_time > 0 else 0
latency = (total_time / total_frames) * 1000 if total_frames > 0 else 0

print(f"\n[4. 실시간 처리 성능 측정]")
print(f" - 총 프레임 수: {total_frames}")
print(f" - 총 소요 시간: {total_time:.2f}초")
print(f" - 평균 FPS: {avg_fps:.2f}")
print(f" - 프레임당 평균 지연 시간: {latency:.2f}ms")

# ----------------------------
# 평가 함수들

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

def compute_accuracy(folder_path, centroids, threshold):
    correct = 0
    total = 0
    files = sorted(os.listdir(folder_path))
    for file in files:
        img = Image.open(os.path.join(folder_path, file)).convert('RGB')
        face = mtcnn(img)
        if face is None:
            continue
        with torch.no_grad():
            emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
        if is_registered(emb, centroids, threshold):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def evaluate_face_recognition_drop():
    acc_before = compute_accuracy("C:/Users/ed007/OneDrive/바탕 화면/before_blur", centroids, THRESHOLD)
    acc_after = compute_accuracy("C:/Users/ed007/OneDrive/바탕 화면/after_blur", centroids, THRESHOLD)
    print(f"\n[2. ArcFace 모자이크 전/후 인식률 비교]")
    print(f" - 원본 얼굴 인식률: {acc_before:.3f}")
    print(f" - 모자이크 후 인식률: {acc_after:.3f}")
    print(f" - 인식률 감소율: {(acc_before - acc_after):.3f}")

def analyze_scenario(gt_labels):
    if np.all(gt_labels == 1):
        return "등록자만"
    elif np.all(gt_labels == 0):
        return "비등록자만"
    else:
        return "혼합"

def smart_evaluation():
    pred_path = "C:/Users/ed007/OneDrive/바탕 화면/pred_labels.npy"
    gt_path = "C:/Users/ed007/OneDrive/바탕 화면/gt_labels.npy"

    pred_labels = np.load(pred_path)

    if os.path.exists(gt_path):
        gt_labels = np.load(gt_path)
        min_len = min(len(gt_labels), len(pred_labels))
        gt_labels = gt_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        scenario = analyze_scenario(gt_labels)

        tp = np.sum((gt_labels == 1) & (pred_labels == 1))
        tn = np.sum((gt_labels == 0) & (pred_labels == 0))
        fp = np.sum((gt_labels == 0) & (pred_labels == 1))
        fn = np.sum((gt_labels == 1) & (pred_labels == 0))
        acc = (tp + tn) / len(gt_labels)

        print("\n[1. 얼굴 탐지 정확도 평가]")
        print(f" - 정탐(TP): {tp}, 오탐(FP): {fp}, 미탐(FN): {fn}, 정음(TN): {tn}")
        print(f" - Accuracy: {acc:.3f}")

        print("\n[상황 판단 및 해석 기준]")
        if scenario == "등록자만":
            print(" - 현재 영상에는 등록자만 등장합니다.")
            print(" - Accuracy는 의미가 없으며, FN(미탐)의 수가 작을수록 좋습니다.")
        elif scenario == "비등록자만":
            print(" - 현재 영상에는 비등록자만 등장합니다.")
            print(" - FP(오탐)의 수가 작을수록 좋으며 Accuracy를 참고할 수 있습니다.")
        else:
            print(" - 등록자와 비등록자가 함께 등장합니다. Accuracy, TP, FN, FP 모두 중요합니다.")
    else:
        print("\n[1. 얼굴 탐지 정확도 평가 생략됨 - GT 없음]")

    evaluate_face_recognition_drop()

# ----------------------------
# 평가 실행
smart_evaluation()
