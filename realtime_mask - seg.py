
import os
import cv2
import numpy as np
import torch
import pytesseract
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ───────────────────────────────────────────────────────────────
# 1) Tesseract OCR 엔진 경로 지정
tess_dir = r'C:\Program Files\Tesseract-OCR'
os.environ['PATH'] = tess_dir + os.pathsep + os.environ.get('PATH', '')
pytesseract.pytesseract.tesseract_cmd = os.path.join(tess_dir, 'tesseract.exe')
_ = pytesseract.get_tesseract_version(cached=False)

# ───────────────────────────────────────────────────────────────
# 2) 타깃 얼굴 임베딩 로드
centroid = np.load(r'C:\Users\ed007\centroid.npy')

# ───────────────────────────────────────────────────────────────
# 3) 모델 초기화
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
seg_model = YOLO('yolov8n-seg.pt')
mtcnn     = MTCNN(
    image_size=224, margin=40,
    thresholds=[0.5, 0.6, 0.7], min_face_size=20,
    keep_all=False, device=device
)
resnet    = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ───────────────────────────────────────────────────────────────
# 4) OCR 설정
OCR_CONFIG = '--oem 3 --psm 7 outputbase digits'

# ───────────────────────────────────────────────────────────────
def mosaic_mask(img, mask, scale=0.05):
    h, w = img.shape[:2]
    small      = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    big_mosaic = cv2.resize(small, (w, h),                     interpolation=cv2.INTER_NEAREST)
    m3 = (mask > 0.5).astype(np.uint8)[:, :, None]
    img[:] = img * (1 - m3) + big_mosaic * m3

# ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")
print("▶ ‘q’ 키로 종료")

THRESHOLD = 0.60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (A) 얼굴 기반 세그멘테이션 + FaceNet 식별 모자이크
    results = seg_model(frame, conf=0.5, iou=0.45)[0]
    if results.masks is not None:
        for box, mask in zip(results.boxes, results.masks.data):
            if int(box.cls) != 0:
                continue
            mask_np = mask.cpu().numpy()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            try:
                face = mtcnn(pil)
                if face is None:
                    raise ValueError
            except:
                mosaic_mask(frame, mask_np)
                continue
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
            cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))
            if cos_sim < THRESHOLD:
                mosaic_mask(frame, mask_np)

    # (B) OCR 기반 숫자 모자이크 + 전처리·디버그
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) 히스토그램 평활화
    gray = cv2.equalizeHist(gray)

    # 2) Adaptive Threshold
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 3) Morphological Closing
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 4) Gaussian Blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    data = pytesseract.image_to_data(
        gray,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    for i in range(len(data['text'])):
        text = str(data['text'][i]).strip()
        if not text:
            continue
        # 안전하게 confidence 변환
        try:
            conf = float(data['conf'][i])
        except:
            conf = 0.0

        # 디버그 로그
        print(f"[OCR] '{text}' ({conf:.0f}%) at {data['left'][i]},{data['top'][i]},{data['width'][i]},{data['height'][i]}")

        # 영역 시각화
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        # 숫자 모자이크
        if text.isdigit() and conf >= 50:
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
            mask[y:y+h, x:x+w] = 1.0
            mosaic_mask(frame, mask)

    # 결과 출력
    cv2.imshow('Mask Faces & Numbers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
