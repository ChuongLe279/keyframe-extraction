from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ACT_THRESHOLD = 2.0

x_col, y_col =[], []
plt.style.use('seaborn-v0_8-darkgrid')  # nicer default look

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], color='#2ecc71', linewidth=2)

ax.set_ylim(0, 10)
ax.set_xlabel("Frame idx")
ax.set_ylabel("ACT score")
ax.set_title("Live ACT Score")
ax.yaxis.set_major_locator(MultipleLocator(0.5))
fig.tight_layout()

#plt.ion()
#plt.show()

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
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

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
accumulated_act = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.resize(cur_frame, (320, 180))

    if prev_frame is not None:
        # Custom window
        cv2.namedWindow('live window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('live window', cur_frame)
        cv2.resizeWindow('live window', 128*5, 72*5)

        act_calculator = ComputeAct(prev_frame, cur_frame, dis)
        act_score = act_calculator.calculate_act()
        accumulated_act += act_score
        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if accumulated_act >= ACT_THRESHOLD:
            accumulated_act = 0

            cv2.namedWindow('keyframe', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('keyframe', frame)
            cv2.resizeWindow('keyframe', 128*5, 72*5)
            cv2.setWindowTitle('keyframe', f'Keyframe - Frame {current_idx}')

        #print(f"Frame idx: {current_idx} | ACT: {act_score}")
        """
        # Draw live graph
        x_col.append(current_idx)
        y_col.append(act_score)
        if current_idx % 3 == 0:
            line.set_data(x_col, y_col)
            ax.set_xlim(x_col[0], x_col[-1])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        """

        cv2.waitKey(1)
        if cv2.getWindowProperty('live window', cv2.WND_PROP_VISIBLE) < 1:
            break

    prev_frame = cur_frame.copy()


cv2.destroyAllWindows()
cap.release()


