# GroundingDINO-TensorRT
This repository contains the codebase for generating ONNX models and [TensorRT](https://github.com/NVIDIA/TensorRT) engines for [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), enabling real-time object detection on NVIDIA hardware.

<p align="center">
  <img src="images/demo_multi.gif" height="400">
</p>

## Guides
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Miscellaneous](#miscellaneous)

## Hardware Requirements
- NVIDIA edge device/PC with NVIDIA GPU

We tested our code on a PC with NVIDIA RTX 4080, Jetson Xavier, Jetson Orin NX, and Jetson AGX Orin. This code should work on any NVIDIA edge device with a proper TensorRT environment setup.
## Installation
- Install Grounding DINO (see [the official repo](https://github.com/IDEA-Research/GroundingDINO) for detailed instructions):
	1. Clone this repo:
		```bash
		git clone https://github.com/shentan-shiina/GroundingDINO-TensorRT.git
		```
	2. Install dependencies:
		```bash
		cd GroundingDINO-TensorRT && pip install -e .
		```
	3. Download pre-trained model weights:
		```bash
		mkdir weights && cd weights
		wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
		cd ..
		```
- If your NVIDIA edge device does not have TensorRT pre-installed, install it via pip:
	```bash
	pip install tensorrt===8.6.1 # We use an old version to avoid unwanted errors, you may try the lastest one.
	```

## Usage
- Generate ONNX model:
	1. Change the specifications in the script (e.g., image size, model path):
		```bash
		gedit scripts/export_onnx.py
		```
	2. Generate precomputed text embeddings:
		```bash
		python scripts/precompute_text_embeddings.py
		```
		You may add your custom text input to `scripts/text_prompt.csv`
	3. Run the script to generate the ONNX model:
		```bash
		python scripts/export_onnx.py
		```
- Test your ONNX model:
	```bash
	python scripts/onnx_inference_video.py
	```
	The raw ONNX model may not be performant, but this usually does not reflect the final TensorRT engine performance.
- Convert to TensorRT engine:
	```bash
	polygraphy convert .asset/groundingdino_400_2_30.onnx \
	 -o .asset/groundingdino_v4_400_2_30_tf32.engine --tf32 \
	 --trt-min-shapes img:[1,3,400,400] encoded_text:[1,2,256] text_token_mask:[1,2] position_ids:[1,2] text_self_attention_masks:[1,2,2]  \
	 --trt-opt-shapes img:[1,3,400,400] encoded_text:[1,4,256] text_token_mask:[1,4] \
	 position_ids:[1,4] text_self_attention_masks:[1,4,4] \
	 --trt-max-shapes img:[1,3,400,400] encoded_text:[1,30,256] text_token_mask:[1,30] position_ids:[1,30] text_self_attention_masks:[1,30,30]
	```
	You may adjust the input shapes above, but ensure the **image size remains fixed**.
- Inference with your TensorRT engine:
	```bash
	python scripts/tensorRT_inference_video.py
	```
	You can directly real-time detect single/multiple objects with any text input.
<p align="center">
  <img src="images/demo_single.gif" height="400">
  <img src="images/demo_multi.gif" height="400">
</p>

You can easily implement our code on your edge device, then run some robot-related task, such as object pick-and-place:
<p align="center">
  <img src="images/demo_piper.gif" height="400">
</p>

## Miscellaneous
1. We precompute the text embeddings to speed up the inference as well as to avoid ONNX conversion issues.
2. To achieve the best performance, we highly recommand using a NVIDIA edge devices (the code can run at 8 FPS at a low-cost Jetson Xavier).
3. You must run the TensorRT engine conversion **on the device you want to implement on** (e.g., you cannot convert the engine on your PC, then copy it to your edge device, which will not work). 
4. Currently, the multi-object detection captioning is a bit messy, we plan to fix it in the future.
5. ...