import numpy as np
import time
from groundingdino.util.inference import annotate
from PIL import Image
import cv2
from threading import Thread
from queue import Queue
from tensorrt_util.tensorrt_gDINO_util import TensorRTInfer,TRT_load_image,TRT_resize_image, TRT_load_data, cal_predict_results

# Preload random image
image = Image.open("./.asset/multi.png")
image_size = (400,400)
text_prompt = "box . charger ."
box_threshold = 0.4
text_threshold = 0.2
input_dict = TRT_load_data(image=image,
                           image_size=image_size,
                           text_prompt=text_prompt)

# Load and init engine
trt_infer = TensorRTInfer("./.asset/groundingdino_v4_400_2_30_b_tf32.engine",input_dict=input_dict)
# Preheat (Essential for text embedding cache loading)
trt_infer.infer()

# Open video file
input_video_path = ".asset/multi.mp4"

cap = cv2.VideoCapture(input_video_path)
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

def detect_objects():
    global last_results
    while True:

        frame = frame_queue.get()
        if frame is None:
            break
        start_time = time.time()

        img_tensor = TRT_resize_image(frame, image_size=image_size)
        model_output = trt_infer.infer_img(img_tensor[None].numpy().astype(np.float32))
        results = cal_predict_results(model_output,
                                       box_threshold,
                                       text_threshold,
                                       text_prompt)
        end_time = time.time()
        print('FPS:', 1 / (time.time() - start_time))

        detection_times.append(end_time - start_time)
        result_queue.put(results)

# Start object detection thread
detection_thread = Thread(target=detect_objects, daemon=True)
detection_thread.start()

frame_count = 0
start_time = time.time()
frame_time = 1 / fps / 2

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
# print(f"Output video saved to: {output_video_path}")
