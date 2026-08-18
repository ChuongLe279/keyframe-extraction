'''High-throughput ACT keyframe extraction; preview is opt-in.'''
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from time import perf_counter
import cv2
import numpy as np

VIDEO_PATH = Path('data/video/1.mp4')
ACT_THRESHOLD = 0.75


class ActComputer:
    '''Persistent, allocation-light ACT calculator.'''

    def __init__(self, shape: tuple[int, int], quality: str = 'turbo') -> None:
        height, width = shape
        preset = cv2.DISOPTICAL_FLOW_PRESET_FAST if quality == 'fast' else cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        self.dis = cv2.DISOpticalFlow.create(preset)
        if quality == 'turbo':
            self.dis.setGradientDescentIterations(8)
            self.dis.setUseSpatialPropagation(False)
        self.warm_start = quality == 'turbo'
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        self.base_map = np.dstack((grid_x, grid_y))
        self.warp_map = np.empty_like(self.base_map)
        self.magnitude = np.empty((height, width), dtype=np.float32)
        self.flow: np.ndarray | None = None

    def calculate(self, previous: np.ndarray, current: np.ndarray) -> float:
        # Turbo warm-starts DIS; other modes retain reference ACT behavior.
        destination = self.flow if self.warm_start else None
        self.flow = self.dis.calc(previous, current, destination)
        cv2.magnitude(self.flow[..., 0], self.flow[..., 1], self.magnitude)
        average_motion = cv2.mean(self.magnitude)[0]
        np.add(self.base_map, self.flow, out=self.warp_map)
        warped = cv2.remap(current, self.warp_map, None, cv2.INTER_LINEAR)
        correlation = float(cv2.matchTemplate(
            previous, warped, cv2.TM_CCORR_NORMED
        )[0, 0])
        return float(np.sqrt(max(0.0, average_motion * (1.0 - correlation))))


class ComputeAct:
    '''Backwards-compatible wrapper around the original public API.'''

    def __init__(self, frame1: np.ndarray, frame2: np.ndarray, dis=None) -> None:
        self.frame1, self.frame2 = frame1, frame2
        self.dis = dis or cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._flow = None

    def find_motion_arrow(self):
        if self._flow is None:
            self._flow = self.dis.calc(self.frame1, self.frame2, None)
        return self._flow

    def measure_arrow_length(self):
        flow = self.find_motion_arrow()
        return cv2.mean(cv2.magnitude(flow[..., 0], flow[..., 1]))[0]

    def calculate_swr(self):
        flow = self.find_motion_arrow()
        height, width = flow.shape[:2]
        map_x, map_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        map_x += flow[..., 0]
        map_y += flow[..., 1]
        warped = cv2.remap(self.frame2, map_x, map_y, cv2.INTER_LINEAR)
        ncc = cv2.matchTemplate(self.frame1, warped, cv2.TM_CCORR_NORMED)[0, 0]
        return 1.0 - float(ncc)

    def calculate_act(self):
        return float(np.sqrt(max(0.0, self.measure_arrow_length() * self.calculate_swr())))


@dataclass(slots=True)
class FramePacket:
    number: int
    gray: np.ndarray
    color: np.ndarray | None


@dataclass(slots=True)
class WorkerFailure:
    error: BaseException


END = object()


def _put_unless_stopped(output, item, stop_event):
    '''Put without leaving a producer permanently blocked during shutdown.'''
    while not stop_event.is_set():
        try:
            output.put(item, timeout=0.1)
            return True
        except Full:
            continue
    return False


def decode_frames(
    cap, output, size, keep_color, frame_step, max_frames, stop_event
):
    '''Decode and preprocess concurrently with optical-flow calculation.'''
    emitted = frame_number = 0
    try:
        while (
            not stop_event.is_set()
            and (max_frames is None or emitted < max_frames)
        ):
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if (frame_number - 1) % frame_step:
                continue
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size,
                interpolation=cv2.INTER_AREA,
            )
            if not _put_unless_stopped(
                output,
                FramePacket(frame_number, gray, frame if keep_color else None),
                stop_event,
            ):
                return
            emitted += 1
    except BaseException as error:
        _put_unless_stopped(output, WorkerFailure(error), stop_event)
    finally:
        _put_unless_stopped(output, END, stop_event)


def write_keyframes(input_queue, output_dir, jpeg_quality):
    '''Encode keyframes without stalling optical flow.'''
    while True:
        item = input_queue.get()
        if item is END:
            return
        number, frame = item
        cv2.imwrite(
            str(output_dir / f'frame_{number:08d}.jpg'), frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )


def extract_keyframes(args) -> list[int]:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {args.video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    keep_color = args.display or args.output_dir is not None
    frame_queue = Queue(maxsize=args.queue_size)
    stop_event = Event()
    producer = Thread(
        target=decode_frames,
        args=(cap, frame_queue, (args.width, args.height), keep_color,
              args.frame_step, args.max_frames, stop_event),
        name='video-decoder',
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_queue = Queue(maxsize=args.queue_size)
        writer = Thread(
            target=write_keyframes,
            args=(write_queue, args.output_dir, args.jpeg_quality), daemon=True,
        )
        writer.start()
    else:
        write_queue = writer = None
    if args.display:
        cv2.namedWindow('live window', cv2.WINDOW_NORMAL)
        cv2.namedWindow('keyframe', cv2.WINDOW_NORMAL)

    computer = ActComputer((args.height, args.width), args.quality)
    previous = None
    accumulated = 0.0
    keyframes = []
    processed = 0
    started = perf_counter()
    producer.start()
    try:
        while True:
            packet = frame_queue.get()
            if packet is END:
                break
            if isinstance(packet, WorkerFailure):
                raise RuntimeError('Video decoder thread failed') from packet.error
            processed += 1
            if previous is not None:
                accumulated += computer.calculate(previous, packet.gray)
                if accumulated >= args.threshold:
                    accumulated = 0.0
                    keyframes.append(packet.number)
                    if args.output_dir is not None:
                        write_queue.put((packet.number, packet.color))
                    if args.display:
                        cv2.imshow('keyframe', packet.color)
                        cv2.setWindowTitle('keyframe', f'Keyframe - Frame {packet.number}')
            if args.display:
                cv2.imshow('live window', packet.color)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            previous = packet.gray
    finally:
        stop_event.set()
        producer.join()
        if writer is not None:
            write_queue.put(END)
            writer.join()
        elapsed = perf_counter() - started
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    rate = processed / elapsed if elapsed else 0.0
    realtime = rate / source_fps if source_fps else 0.0
    print(f'Processed {processed:,}/{total:,} sampled frames in {elapsed:.2f}s '
          f'({rate:.1f} FPS, {realtime:.1f}x realtime); '
          f'found {len(keyframes):,} keyframes.')
    if args.print_keyframes and keyframes:
        print('Keyframe frame numbers:', ', '.join(map(str, keyframes)))
    return keyframes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', nargs='?', type=Path, default=VIDEO_PATH)
    parser.add_argument('--threshold', type=float, default=ACT_THRESHOLD)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--quality', choices=('turbo', 'ultrafast', 'fast'), default='turbo')
    parser.add_argument('--frame-step', type=int, default=1, help='process every Nth decoded frame')
    parser.add_argument('--queue-size', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, help='save selected frames as JPEGs')
    parser.add_argument('--jpeg-quality', type=int, default=90)
    parser.add_argument('--display', action='store_true', help='show slower live preview')
    parser.add_argument('--print-keyframes', action='store_true')
    parser.add_argument('--max-frames', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.width, args.height, args.frame_step, args.queue_size) < 1:
        parser.error('width, height, frame-step, and queue-size must be positive')
    return args


if __name__ == '__main__':
    cv2.setUseOptimized(True)
    extract_keyframes(parse_args())
