from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import threading
import time
import csv

ACT_THRESHOLD = 0.95
VIDEO_PATH = Path("./data/video/1.mp4")
OUTPUT_DIR = Path("./data/keyframes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def extracting_keyframes(VIDEO_PATH, start_idx, end_idx, cap, dis, output_dir):
    # Seek to the starting frame instead of reading from 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    prev_frame = None
    accumulated_act = 0
    
    while True:
        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_idx >= end_idx:
            break

        ret, frame = cap.read()
        if not ret:
            break

        cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.resize(cur_frame, (320, 180))

        if prev_frame is not None:
            act_calculator = ComputeAct(prev_frame, cur_frame, dis)
            act_score = act_calculator.calculate_act()
            accumulated_act += act_score
            current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if accumulated_act >= ACT_THRESHOLD:
                accumulated_act = 0
                """save frame"""
                out_path = Path(output_dir) / f"frame_{current_idx:06d}.jpg"
                print(f"Saved frame_{current_idx:06d}.jpg")
                cv2.imwrite(str(out_path), frame)

        prev_frame = cur_frame.copy()

    cap.release()


def extracting_videos_with_threads(VIDEO_PATH, num_threads, OUTPUT_DIR):
     # Open the video file
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Error: could not open video file")
        exit()
    else:
        print("Open video.")

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    boundaries  = np.linspace(0, totalFrames, num=num_threads + 1, dtype=int)
    threads = []
    for i in range(num_threads):
        start_idx = int(boundaries[i])
        end_idx = int(boundaries[i + 1])

        thread_cap = cv2.VideoCapture(str(VIDEO_PATH))

        # Initialize DIS Optical Flow 
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        t = threading.Thread(
            target=extracting_keyframes,
            args=(VIDEO_PATH, start_idx, end_idx, thread_cap, dis, OUTPUT_DIR),
            name=f"Thread-{i}"
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def delete_folder(FOLDER_PATH):
    for file_path in FOLDER_PATH.glob("*.jpg"):
        file_path.unlink()

if __name__ == "__main__":
    lowest_time = np.inf
    runtime_log = []
    print("start")

    min_thread = 4
    max_thread = 12

    current_thread = min_thread
    best_thread = None
    while current_thread <= max_thread:
        start_time = time.perf_counter()
        extracting_videos_with_threads(VIDEO_PATH, current_thread, OUTPUT_DIR)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        runtime_log.append(execution_time)

        if lowest_time > execution_time:
            lowest_time = execution_time
            best_thread = current_thread
        current_thread += 1

        delete_folder(OUTPUT_DIR)

    print("done")
    print(f"Best thread: {best_thread} | Lowest time: {lowest_time}")
    print(f"Execution time: {runtime_log}")

    # 10 thread da best
   


