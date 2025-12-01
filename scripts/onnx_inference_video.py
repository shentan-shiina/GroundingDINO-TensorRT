import numpy as np
import torch
from PIL import Image
from threading import Thread
from queue import Queue
from groundingdino.util.inference import load_model, predict_onnx, annotate
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_text_dict_cache_path
from tensorrt_util.tensorrt_gDINO_util import TensorRTInfer,TRT_load_image,TRT_resize_image, TRT_load_data, cal_predict_results
import cv2
import time
import os
import onnxruntime as ort
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
# Load model
onnx_model_path = './.asset/groundingdino_v4_350_2_30_b.onnx'
session = ort.InferenceSession(onnx_model_path)

TEXT_PROMPT = "cat ."
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.2
image_size = (600,600)
is_load_text_embeddings = True

cache_path = get_text_dict_cache_path(TEXT_PROMPT)
predict_cache_path = get_text_dict_cache_path(preprocess_caption(caption=TEXT_PROMPT),predict=True)
if is_load_text_embeddings and os.path.exists(cache_path) and os.path.exists(predict_cache_path):
    text_dict = torch.load(cache_path, map_location='cuda')
    pred_phrase = torch.load(predict_cache_path, map_location='cuda')

# Open video file
input_video_path = "./.asset/multi-cats.mp4"

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define label colors
label_colors = {}

def get_label_color(label):
    if label not in label_colors:
        label_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
    return label_colors[label]

# Queue for frame processing
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
last_results = None  # Store the last detected results

detection_times = []  # Store detection time per frame

transform = T.Compose(
    [
        T.RandomResize([300], max_size=300),
        T.ToTensor(),
    ]
)

def detect_objects():
    global last_results
    while True:

        frame = frame_queue.get()
        if frame is None:
            break

        start_time = time.time()
        frame = Image.fromarray(frame).convert("RGB")
        image_transformed = TRT_resize_image(frame, image_size=image_size)
        if is_load_text_embeddings:
            text_dict_loop = {k: v.clone() for k, v in text_dict.items()}
            pred_phrase_loop = pred_phrase
        else:
            text_dict_loop = None
            pred_phrase_loop = None

        results = predict_onnx(
            session=session,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            load_text_embeddings=is_load_text_embeddings,
            text_dict=text_dict_loop,
            pred_phrase=pred_phrase_loop,
        )

        end_time = time.time()
        print('Inference Time:', time.time() - start_time)

        detection_times.append(end_time - start_time)

        result_queue.put(results)

# Start object detection thread
detection_thread = Thread(target=detect_objects, daemon=True)
detection_thread.start()

frame_count = 0
start_time = time.time()
frame_time = 1 / fps / 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    start_frame_time = time.time()
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    if not result_queue.empty():
        last_results = result_queue.get()

    if last_results is not None:
        boxes, logits, phrases = last_results
        frame = annotate(image_source=np.asarray(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))), boxes=boxes, logits=logits, phrases=phrases)

    cv2.imshow("Object Detection", frame)

    elapsed_time = time.time() - start_frame_time
    sleep_time = max(0, frame_time - elapsed_time)  # Avoid negative sleep time
    time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame_queue.put(None)  # Stop the detection thread
cap.release()
cv2.destroyAllWindows()

# Calculate FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time
avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0

print(f"Total frames processed: {frame_count}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average FPS: {fps:.2f}")
print(f"Average model inference time per frame: {avg_detection_time:.4f} seconds")

