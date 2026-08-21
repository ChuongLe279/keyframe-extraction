import cv2
from pathlib import Path

video_path = Path("./data/video/1.mp4")
cap = cv2.VideoCapture(video_path)

# 1. Define your frame boundaries (zero-based indexing)
start_frame = 100
end_frame = 1000

# Optional: Verify video bounds
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end_frame = min(end_frame, total_frames - 1)

# 2. Seek directly to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 3. Loop through the specific segment
current_frame = start_frame
while current_frame <= end_frame:
    success, frame = cap.read()
    
    # Break early if the video ends unexpectedly
    if not success:
        print("Reached end of video stream early.")
        break

    cv2.imshow("Segment", frame)
        

    
    current_frame += 1

# 4. Clean up resources
cap.release()
cv2.destroyAllWindows()