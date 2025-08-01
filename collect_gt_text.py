# collect_gt_texts.py
import cv2
import numpy as np
import easyocr
import re

# 민감 텍스트 판별 함수
def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}', r'01[0-9]-\d{3,4}-\d{4}',
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        r'\d{1,4}동|\d{1,4}호', r'[\uac00-\ud7a3]+[시군구동읍면로길]',
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|오피스텔|연립)',
        r'\d{1,4}-\d{1,4}', r'\d{2,4}-\d{2,4}-\d{4,7}', r'\d{9,14}',
        r'(대학교|중학교|고등학교|회사|직장|소속)', r'(이름|성명)[:：]?\s?[가-힣]{2,4}'
    ]
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# OCR 초기화
ocr = easyocr.Reader(['ko', 'en'])

# 웹캠 연결
cap = cv2.VideoCapture(0)
gt_texts = []
print("▶ GT 텍스트 수집 중... (q 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    has_sensitive = 0
    texts = ocr.readtext(frame)
    for (_, text, _) in texts:
        if is_sensitive_text(text):
            has_sensitive = 1
            break

    gt_texts.append(has_sensitive)

    # 프레임 보여주기
    cv2.imshow("GT 수집", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 저장
gt_texts = np.array(gt_texts)
np.save("gt_texts.npy", gt_texts)
print(f"▶ 완료: 총 {len(gt_texts)} 프레임 중 {np.sum(gt_texts)}개에서 민감 텍스트 감지")
