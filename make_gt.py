import cv2
import numpy as np
import os

# --- 설정 ---
video_path = 'your_video.mp4'  # 등록자와 비등록자 함께 있는 영상
save_dir = './'  # 저장 경로
gt_label_path = os.path.join(save_dir, 'gt_labels.npy')
gt_bbox_path = os.path.join(save_dir, 'gt_bboxes.npy')

# --- 전역 변수 ---
clicks = []
bboxes = []
gt_labels = []

# --- 마우스 콜백 함수 ---
def click_event(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        if len(clicks) == 2:
            print(f"[선택 완료] bbox: {clicks}")
            cv2.rectangle(param, clicks[0], clicks[1], (0, 255, 0), 2)
            cv2.imshow('Frame', param)

# --- 비디오 실행 ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ 영상 파일을 열 수 없습니다.")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"\n[프레임 {frame_idx}] 등록자 얼굴을 선택해주세요.")
    clone = frame.copy()
    clicks = []

    cv2.imshow('Frame', clone)
    cv2.setMouseCallback('Frame', click_event, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter 키 → 등록자 있음
            if len(clicks) == 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                bboxes.append([x1, y1, x2, y2])
                gt_labels.append(1)
                break
            else:
                print("⚠️ 두 점을 모두 클릭해주세요.")
        elif key == ord('n'):  # n 키 → 등록자 없음
            bboxes.append([0, 0, 0, 0])
            gt_labels.append(0)
            print("[스킵] 등록자 없음으로 처리됨.")
            break
        elif key == ord('q'):  # 종료
            cap.release()
            cv2.destroyAllWindows()
            np.save(gt_label_path, np.array(gt_labels))
            np.save(gt_bbox_path, np.array(bboxes))
            print(f"\n✅ 저장 완료 → {gt_label_path}, {gt_bbox_path}")
            exit()

    frame_idx += 1

# --- 저장 ---
cap.release()
cv2.destroyAllWindows()
np.save(gt_label_path, np.array(gt_labels))
np.save(gt_bbox_path, np.array(bboxes))
print(f"\n✅ 전체 저장 완료 → {gt_label_path}, {gt_bbox_path}")
