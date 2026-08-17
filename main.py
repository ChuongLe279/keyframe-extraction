from pathlib import Path
import cv2

VIDEO_PATH = Path("./data/video/1.mp4")

if VIDEO_PATH.exists():
    print("Found video.")
else:
    print("Not found video. Check path")

# Open the video file
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Error: could not open video file")
    exit()
else:
    print("Open video.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width = {frame_width} | Frame height = {frame_height}")

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('LIVE', gray_frame)
    # Custom window
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('custom window', gray_frame)
    cv2.resizeWindow('custom window', 200, 200)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


