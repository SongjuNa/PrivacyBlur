import os
# 1) Tesseract 설치 폴더를 PATH에 추가
tess_dir = r'C:\Program Files\Tesseract-OCR'
os.environ['PATH'] = tess_dir + os.pathsep + os.environ.get('PATH','')

import pytesseract
# 2) pytesseract에 명시적 실행 파일 경로 지정
pytesseract.pytesseract.tesseract_cmd = os.path.join(tess_dir, 'tesseract.exe')
# 3) 캐시 초기화
_ = pytesseract.get_tesseract_version(cached=False)
import cv2
import numpy as np
import torch
import pytesseract
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 0) 타깃 임베딩 로드
centroid = np.load(r'C:\Users\ed007\centroid.npy')

# 1) 모델 초기화
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
seg_model = YOLO('yolov8n-seg.pt')
mtcnn     = MTCNN(
    image_size=224, margin=40,
    thresholds=[0.5, 0.6, 0.7], min_face_size=20,
    keep_all=False, device=device
)
resnet    = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 2) OCR 설정 (숫자 전용)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'  # 시스템에 맞게 조정
OCR_CONFIG = '--oem 3 --psm 6 outputbase digits'

# 3) 모자이크 함수 (mask: HxW float 0~1)
def mosaic_mask(img, mask, scale=0.05):
    h, w = img.shape[:2]
    small      = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    big_mosaic = cv2.resize(small, (w, h),                     interpolation=cv2.INTER_NEAREST)
    m3 = (mask > 0.5).astype(np.uint8)[:, :, None]
    img[:] = img * (1 - m3) + big_mosaic * m3

# 4) 웹캠 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")
print("▶ ‘q’ 키로 종료")

THRESHOLD = 0.60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (A) 얼굴 기반 세그멘테이션 마스크 모자이크
    results = seg_model(frame, conf=0.5, iou=0.45)[0]

    # masks가 있을 때만 처리
    if results.masks is not None:
        for box, mask in zip(results.boxes, results.masks.data):
            # 사람 클래스만 처리 (YOLO 클래스 ID 0)
            if int(box.cls) != 0:
                continue

            mask_np = mask.cpu().numpy()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            # 얼굴 인식 시도
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            face = None
            try:
                face = mtcnn(pil)
            except:
                pass

            # 얼굴 인식 실패 시 무조건 모자이크
            if face is None:
                mosaic_mask(frame, mask_np)
                continue

            # 임베딩 비교
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
            cos_sim = np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb))

            # 임계치 아래면 모자이크
            if cos_sim < THRESHOLD:
                mosaic_mask(frame, mask_np)

    # (B) OCR 기반 숫자 영역 모자이크
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, config=OCR_CONFIG, output_type=pytesseract.Output.DICT)
    for i, text in enumerate(data['text']):
        if not text.strip().isdigit():
            continue
        conf = float(data['conf'][i])
        if conf < 50:
            continue
        x, y, w, h = (data['left'][i], data['top'][i],
                      data['width'][i], data['height'][i])
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        mask[y:y+h, x:x+w] = 1.0
        mosaic_mask(frame, mask)

    # 결과 출력
    cv2.imshow('Mask Faces & Numbers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
