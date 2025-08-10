import cv2
import numpy as np
import re
import easyocr
import os
import time

# ----------------------------
# 설정
video_path = r"C:/Users/User/Downloads/text_video2.mp4"  # 영상 경로
gt_path = r"C:/Users/User/Downloads/gt_texts2.npy"       # GT 텍스트 레이블
pred_save_path = r"C:/Users/User/Downloads/pred_texts2.npy"  # 탐지 결과 저장 경로

# 민감 텍스트 패턴 정의
def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}',                             # 주민등록번호
        r'01[0-9]-\d{3,4}-\d{4}',                   # 휴대폰 번호
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  # 이메일 주소
        r'(이름|성명)[:：]?\s?[가-힣]{2,4}',         # 이름
        r'\d{2,4}-\d{2,4}-\d{4,7}',                 # 계좌번호
        r'\d{9,14}',                                 # 사업자 번호 등
        r'(중학교|고등학교|대학교|회사|직장|소속)',   # 소속
        r'\d{1,4}동|\d{1,4}호',                      # 동/호수
        r'[가-힣]{1,20}(시|군|구|동|읍|면|로|길)(\s|[0-9]|$)',        # 주소
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|오피스텔|연립)',  # 건물
        r'\d{1,4}-\d{1,4}',                          # 지번
        r'(서울|부산|대구|인천|광주|대전|울산|세종|제주|경기|강원|충북|충남|전북|전남|경북|경남|창원)(특별시|광역시|특례시|도|특별자치도)'
    ]
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# ----------------------------
# 영상 처리 및 민감 텍스트 탐지
print("▶ 영상 처리 시작...")
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['ko', 'en'])
pred_texts = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    print(f"\n▶ 처리 중인 프레임: {frame_idx}")
    
    texts = reader.readtext(frame)

    pred_flag = 0
    for _, text, conf in texts:
        if is_sensitive_text(text):
            print(f"[탐지] 민감 텍스트: '{text}' (conf: {conf:.2f})")
            pred_flag = 1
            break
    pred_texts.append(pred_flag)

cap.release()
pred_texts = np.array(pred_texts)
np.save(pred_save_path, pred_texts)
print(f"▶ 예측 결과 저장 완료: {len(pred_texts)} 프레임 → pred_texts.npy")

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
