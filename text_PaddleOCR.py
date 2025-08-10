import cv2
import numpy as np
import re
import os
import time
from paddleocr import PaddleOCR

# ----------------------------
# 설정
video_path = r"C:/Users/User/Downloads/text_night_video3.mp4"  # 영상 경로
gt_path = r"C:/Users/User/Downloads/gt_texts2.npy"       # GT 텍스트 레이블
pred_save_path = r"C:/Users/User/Downloads/pred_texts2.npy"  # 탐지 결과 저장 경로

# ----------------------------
# 프레임 전처리 함수 (야간 대응)
def preprocess_frame(frame):
    """야간 환경에서 텍스트 인식 성능 향상을 위한 전처리"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #그레이스케일
    #국소 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 감마 보정 (어두운 부분 밝게)
    def adjust_gamma(image, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    gamma_corrected = adjust_gamma(enhanced, gamma=1.8)

    # 4. 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gamma_corrected, h=15)

    # 5. 샤프닝 (텍스트 윤곽 강조)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # 6. 밝기/대비 보정
    final = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=40)

    # 7. BGR 변환 (PaddleOCR 입력은 3채널 필요)
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    return result
    return result

# ----------------------------
# 민감 텍스트 판별 함수 정의
def is_sensitive_text(text):
    text_no_space = text.replace(" ", "")  # OCR 결과 공백 제거

    patterns = [
        r'\d{6}-\d{7}',                             # 주민등록번호
        r'01[0-9]-\d{3,4}-\d{4}',                   # 휴대폰 번호
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  # 이메일 주소
        r'(이름|성명)[:：]?\s?[가-힣]{2,4}',         # 이름
        r'\d{2,4}-\d{2,4}-\d{4,7}',                 # 계좌번호
        r'\d{9,14}',                                # 사업자 번호 등
        r'(중학교|고등학교|대학교|회사|직장|소속)',   # 소속
        r'\d{1,4}동|\d{1,4}호',                      # 동/호수
        r'[가-힣]{1,20}(아파트|빌라|주택|오피스텔|연립)',  # 건물명
        r'\d{1,4}-\d{1,4}',                          # 지번
        r'(서울|부산|대구|인천|광주|대전|울산|세종|제주)(특별시|광역시|특별자치도|도)', #지역명
        r'[가-힣]+(시|군|구)',                        # 시/군/구
        r'[가-힣0-9]+(동|읍|면)',                    # 상세 주소
        r'[가-힣0-9\-\.]+\s?(로|길)'                 # 도로명 주소
    ]

    for p in patterns:
        if re.search(p, text_no_space):
            return True

    return False

# ----------------------------
# PaddleOCR 초기화 (한글+영어, GPU 사용)
print("▶ PaddleOCR 초기화 중...")
ocr = PaddleOCR(
    lang='korean',
    use_textline_orientation=True                
)

# ----------------------------
# 영상 처리 및 민감 텍스트 탐지
print("▶ 영상 처리 시작...")
cap = cv2.VideoCapture(video_path)
pred_texts = []  # 예측 결과 저장 리스트

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    print(f"\n▶ 처리 중인 프레임: {frame_idx}")

    # 프레임 전처리 (야간 대비 향상)
    processed_frame = preprocess_frame(frame)

    # PaddleOCR 인식 (이미지 배열 직접 전달 가능)
    result = ocr.predict(processed_frame)
    pred_flag = 0
    
    # PaddleOCR 결과 구조: [[bbox, (text, confidence)], ...]
    for line in result:
        rec_texts = line.get('rec_texts', [])
        rec_scores = line.get('rec_scores', [])
        
        for text, conf in zip(rec_texts, rec_scores):
            print(f"인식 텍스트: {text} (신뢰도: {conf:.2f})")
            if conf > 0.5 and is_sensitive_text(text):
                print(f"[탐지] 민감 텍스트: '{text}' (conf: {conf:.2f})")
                pred_flag = 1
                break
        if pred_flag == 1:
            break
    pred_texts.append(pred_flag)

cap.release()
pred_texts = np.array(pred_texts)
np.save(pred_save_path, pred_texts)
print(f"▶ 예측 결과 저장 완료: {len(pred_texts)} 프레임 → {pred_save_path}")

# ----------------------------
# 성능 평가 함수
def evaluate_text_detection(gt_texts, pred_texts):
    gt_texts = np.array(gt_texts)
    pred_texts = np.array(pred_texts)

    # 길이 맞추기
    min_len = min(len(gt_texts), len(pred_texts))
    gt_texts = gt_texts[:min_len]
    pred_texts = pred_texts[:min_len]

    tp = np.sum((gt_texts == 1) & (pred_texts == 1))
    tn = np.sum((gt_texts == 0) & (pred_texts == 0))
    fp = np.sum((gt_texts == 0) & (pred_texts == 1))
    fn = np.sum((gt_texts == 1) & (pred_texts == 0))

    acc = (tp + tn) / len(gt_texts) if len(gt_texts) > 0 else 0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n[3. 민감 텍스트 탐지 정확도]")
    print(f" - 정탐(TP): {tp}, 오탐(FP): {fp}, 미탐(FN): {fn}, 정음(TN): {tn}")
    print(f" - Accuracy : {acc:.3f}")
    print(f" - Precision: {precision:.3f}")
    print(f" - Recall   : {recall:.3f}")
    print(f" - F1 Score : {f1:.3f}")

# ----------------------------
# 평가 실행
if os.path.exists(gt_path):
    gt = np.load(gt_path)
    pred = np.load(pred_save_path)
    evaluate_text_detection(gt, pred)
else:
    print(f"[오류] GT 파일이 존재하지 않습니다: {gt_path}")
