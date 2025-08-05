import cv2
import os

# 🎥 1. 영상 경로 설정 (여기에 너의 영상 파일 경로 넣어줘)
video_path = r"C:/Users/ed007/OneDrive/바탕 화면/your_video.mp4"

# 📁 2. 프레임 저장 경로 설정
save_dir = r"C:/Users/ed007/OneDrive/바탕 화면/frames"
os.makedirs(save_dir, exist_ok=True)

# 🎞️ 3. 영상 열기
cap = cv2.VideoCapture(video_path)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 파일명: frame_000.jpg, frame_001.jpg ...
    filename = f"frame_{frame_idx:03d}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # 저장
    cv2.imwrite(filepath, frame)

    frame_idx += 1

cap.release()
print(f"✅ 총 {frame_idx}개 프레임 저장 완료 → {save_dir}")
