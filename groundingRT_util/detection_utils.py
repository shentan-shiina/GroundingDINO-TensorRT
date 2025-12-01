import cv2
import time
import numpy as np
from threading import Thread
from PIL import Image as PILImage
from queue import Queue
import pyrealsense2 as rs
from groundingdino.util.inference import annotate_xyxy
from groundingRT_util.realsense_manager import RealSenseManager_v2
from tensorrt_util.tensorrt_gDINO_util import (
    TensorRTInfer, TRT_resize_image, TRT_load_data, cal_predict_results
)

class ObjectDetector:
    def __init__(self, config):
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.last_results = None
        self.config = config
        self._init_model()

    def _init_model(self):
        dummy_image = PILImage.new("RGB", self.config['image_size'])
        input_dict = TRT_load_data(dummy_image, self.config['image_size'], self.config['text_prompt'])
        self.trt_infer = TensorRTInfer(
            self.config['model_path'],
            input_dict
        )
        self.trt_infer.infer()  # Warmup

    def detect_objects(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break

            start = time.time()
            img_tensor = TRT_resize_image(frame, self.config['image_size'])
            model_output = self.trt_infer.infer_img(img_tensor[None].numpy().astype(np.float32))
            results = cal_predict_results(
                model_output,
                self.config['box_threshold'],
                self.config['text_threshold'],
                self.config['text_prompt']
            )
            self.result_queue.put(results)
            print(f"FPS: {1 / (time.time() - start):.1f}")

class ObjectDetection:
    def __init__(self, config):
        self.config = config
        self.visualizer = DetectionVisualizer()
        self.camera = RealSenseManager_v2(config)
        self.detector = ObjectDetector(config)
        self._init_threads()

    def _init_threads(self):
        self.detection_thread = Thread(
            target=self.detector.detect_objects,
            daemon=True
        )
        self.detection_thread.start()

    def run(self):
        try:
            while True:
                depth_frame, color_frame = self.camera.get_frames()
                if not depth_frame or not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                frame, _ = self._process_frame(color_img, depth_frame)
                if self.config['is_display']:
                    self._visualize_frame(depth_frame, frame)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            self.shutdown()

    def _process_frame(self, color_img, depth_frame):
        # Detection processing
        resized = cv2.resize(color_img, self.config['image_size'])
        if self.detector.frame_queue.empty():
            self.detector.frame_queue.put(resized.copy())

        # Get results
        if not self.detector.result_queue.empty():
            self.detector.last_results = self.detector.result_queue.get()

        frame, xxyy = self.visualizer.process_detections(
            self.detector.last_results,
            depth_frame,
            color_img,
            self.config['top_1'],
            return_xyxy=True
        )
        return frame, xxyy

    def _visualize_frame(self, depth_frame, frame):
        # Show combined view
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        combined = np.hstack((frame, depth_colormap))
        cv2.imshow("Detection + Depth", combined)

    def shutdown(self):
        self.camera.stop()
        self.detector.frame_queue.put(None)
        self.detection_thread.join()
        cv2.destroyAllWindows()


class DetectionVisualizer:
    def __init__(self):
        self.label_colors = {}

    def get_label_color(self, label):
        if label not in self.label_colors:
            self.label_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.label_colors[label]

    def process_detections(self, results, depth_frame, frame, top_1=False, return_xyxy=False):
        if not results or len(results[2]) == 0:
            if return_xyxy:
                return frame, None
            else:
                return frame


        boxes, logits, phrases = results

        if top_1 and len(logits) > 1:
            max_idx = np.argmax(logits)
            boxes = boxes[max_idx:max_idx + 1]
            logits = logits[max_idx:max_idx + 1]
            phrases = phrases[max_idx:max_idx + 1]

        frame, xyxy = annotate_xyxy(
            image_source=np.asarray(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))),
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )

        for (x1, y1, x2, y2), phrase in zip(np.int32(xyxy), phrases):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth = depth_frame.get_distance(cx, cy)

            if depth > 0:
                label = f"{depth:.2f}m"
                color = self.get_label_color(phrase)
                cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
                cv2.putText(frame, label, (x2, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if return_xyxy:
            return frame, xyxy
        else:
            return frame