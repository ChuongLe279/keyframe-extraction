from pathlib import Path
import cv2
import numpy as np


VIDEO_PATH = Path("./data/video/1.mp4")

if VIDEO_PATH.exists():
    print("Found video.")
else:
    print("Not found video. Check path")

class ComputeAct:
    def __init__(self, frame1, frame2, dis) -> None:
        self.frame1 = frame1
        self.frame2 = frame2
        self.dis = dis
        self._flow = None  # cache

    def find_motion_arrow(self):
        if self._flow is None:
            self._flow = self.dis.calc(self.frame1, self.frame2, None)
        return self._flow

    def measure_arrow_length(self):
        flow = self.find_motion_arrow()
        dx = flow[..., 0]
        dy = flow[..., 1]
        return np.mean(np.sqrt(dx**2 + dy**2))

    def calculate_swr(self):
        flow = self.find_motion_arrow()
        height, width = flow.shape[:2]

        map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = (map_x + flow[..., 0]).astype(np.float32)
        map_y = (map_y + flow[..., 1]).astype(np.float32)

        warped_J = cv2.remap(self.frame2, map_x, map_y, cv2.INTER_LINEAR)
        ncc = cv2.matchTemplate(self.frame1, warped_J, cv2.TM_CCORR_NORMED)[0, 0]
        return 1.0 - ncc

    def calculate_act(self):
        amm = self.measure_arrow_length()
        swr = self.calculate_swr()
        return np.sqrt(amm * swr)
    

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

# Initialize DIS Optical Flow 
dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
prev_frame = None

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Custom window
        cv2.namedWindow('live window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('live window', gray_frame)
        cv2.resizeWindow('live window', 128*5, 72*5)



    prev_frame = gray_frame.copy()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


