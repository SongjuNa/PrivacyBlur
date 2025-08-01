# collect_gt_texts.py
import cv2
import numpy as np
import easyocr
import re

# 민감 텍스트 판별 함수
def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}',                         #  주민등록번호 (예: 900101-1234567)
    r'01[0-9]-\d{3,4}-\d{4}',              #  휴대폰 번호 (예: 010-1234-5678)
    
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  #  이메일 주소
    
    r'\d{1,4}동|\d{1,4}호',                #  아파트 동/호수 (예: 302동, 1502호)
    r'[\uac00-\ud7a3]+[시군구동읍면로길]', #  도로명 주소 일부 (예: 강남구, 서초동, 종로길)
    
    r'\d{9,14}',                           #  사업자 등록번호 / 계좌번호 / 전화번호 등 (9~14자리 숫자열)
    r'(대학교|중학교|고등학교|회사|직장|소속)', #  소속기관 (학교, 회사 등)
    
    r'(이름|성명)[:：]?\s?[가-힣]{2,4}' ,    #  이름 표기 (예: 이름: 김민수, 성명：이영희)
        r'[\uac00-\ud7a3]+[시군구동읍면로길]',         # 지역 주소 단어
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|맨션|오피스텔|연립)',  # 건물 유형
        r'\d{1,4}-\d{1,4}',                            # 지번 주소

        r'\d{2,4}-\d{2,4}-\d{4,7}',                    # 일반 은행형 계좌번호
        r'\d{9,14}'                                # 숫자만 있는 계좌 
    ]
    for p in patterns:
        if re.search(p, text.replace(" ", "")):
            return True
    return False

# OCR 초기화
ocr = easyocr.Reader(['ko', 'en'])

# 웹캠 연결
camera_index=0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")
    
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
