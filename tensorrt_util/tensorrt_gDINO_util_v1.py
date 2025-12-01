import numpy as np
import time
from groundingdino.util.inference import load_image, annotate
from groundingdino.util.utils import get_text_dict_cache_path
import torch
import os
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from groundingdino.util.inference import load_image, annotate
from groundingdino.util.utils import get_text_dict_cache_path, preprocess_caption
import torch
import os
from PIL import Image
import cv2
import groundingdino.datasets.transforms as T
import tensorrt_util.common as common
from cuda import cudart
from torchvision import transforms
import bisect
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import load_model
## Pre-loading -------------------------------------
class TensorRTInfer:
    """
    Implements inference for the GroundingDINO TensorRT engine.
    """

    def __init__(self, engine_path, input_dict = None):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.batch_size = 1
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Set context profile shape to deal with dynamic shapes
        if input_dict is not None:
            self.context.set_input_shape("img", input_dict['img'].shape)
            self.context.set_input_shape("encoded_text", input_dict['encoded_text'].shape)
            self.context.set_input_shape("text_token_mask", input_dict['text_token_mask'].shape)
            self.context.set_input_shape("position_ids", input_dict['position_ids'].shape)
            self.context.set_input_shape("text_self_attention_masks", input_dict['text_self_attention_masks'].shape)
            self.inputs_dict = input_dict

        # self.init_allocate()
        (self.input_buffer,
         self.output_buffer,
         self.bindings,
         self.stream) = self.init_allocate_v2()

    def init_allocate(self):
        # Setup I/O bindings
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_input_shape(name, profile_shape[2])
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = trt.volume(shape)
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "Allocated {} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def init_allocate_v2(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            binding = self.engine.get_tensor_name(i)
            # size = trt.volume(engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = self.context.get_tensor_shape(binding)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self):
        """
        Execute inference on a batch of images.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        # Refresh both image & text embedding buffer, for preheat/reload
        for i, (_, value) in enumerate(self.inputs_dict.items()):
            common.memcpy_host_to_device(self.inputs[i]["allocation"], value)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]

    def infer_v2(self):
        """
        Execute inference on a batch of images.
        :return A list of outputs as numpy arrays.
        """
        for i, (key, value) in enumerate(self.inputs_dict.items()):
            self.input_buffer[i][0][:] = value.flatten()  # Copy data into the input buffer
        for inp in self.input_buffer:
            cuda.memcpy_htod_async(inp[1], inp[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.output_buffer:
            cuda.memcpy_dtoh_async(out[0], out[1], self.stream)
        self.stream.synchronize()
        return [out[0] for out in self.output_buffer]

    def infer_img_v2(self, img):
        """
        Execute inference on a batch of images.
        :return A list of outputs as numpy arrays.
        """
        self.input_buffer[0][0][:] = img.flatten()  # Copy data into the input buffer
        cuda.memcpy_htod_async(self.input_buffer[0][1], self.input_buffer[0][0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.output_buffer:
            cuda.memcpy_dtoh_async(out[0], out[1], self.stream)
        self.stream.synchronize()
        return [out[0] for out in self.output_buffer]

    def infer_img(self, img):
        """
        Execute inference on a batch of images.
        :param img: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        # Only refresh image buffer
        common.memcpy_host_to_device(self.inputs[0]["allocation"], img)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]

## Pre-processing ----------------------------------

def GDINO_TRT_load_data(image_path, image_size, text_prompt):
    image_source, image = load_image(image_path)
    image_resized,_ = T.resize(image, target=None, size=image_size)

    cache_path = get_text_dict_cache_path(text_prompt)
    if os.path.exists(cache_path):
        text_dict = torch.load(cache_path, map_location='cpu')
    else:
        print('Text embedding cache not found! ', cache_path)

    encoded_text = text_dict["encoded_text"]
    text_token_mask = text_dict["text_token_mask"]
    position_ids = text_dict["position_ids"]
    text_self_attention_masks = text_dict["text_self_attention_masks"]

    inputs_dict = {
        "img": image_resized[None].cpu().numpy().astype(np.float32),
        "encoded_text": encoded_text.cpu().numpy().astype(np.float32),
        "text_token_mask": text_token_mask.cpu().numpy().astype(np.bool_),
        "position_ids": position_ids.cpu().numpy().astype(np.int32),
        "text_self_attention_masks": text_self_attention_masks.cpu().numpy().astype(np.bool_),
    }
    return inputs_dict

def TRT_load_image(image_path, image_size):
    image_source, image = load_image(image_path)
    image_resized,_ = T.resize(image, target=None, size=image_size)
    return image_resized[None].numpy().astype(np.float32), image_source

# def numpy_resize_image(image, image_size):
#     if isinstance(image, Image.Image):
#         image_resized = np.transpose(np.asarray(image.resize(image_size)), (2, 1, 0))
#     else:
#         image_resized = np.transpose(cv2.resize(image,image_size), (2, 1, 0))
#     return image_resized

def TRT_resize_image(image, image_size):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # if isinstance(image, Image.Image):
    #     image = np.asarray(image)
    image_tensor, _ = transform(image, None)
    image_resized, _ = T.resize(image_tensor, target=None, size=image_size)
    return image_resized

def TRT_load_text_embedding(text_prompt):
    cache_path = get_text_dict_cache_path(text_prompt)
    if os.path.exists(cache_path):
        text_dict = torch.load(cache_path, map_location='cpu')
    else:
        print('Text embedding cache not found! ', cache_path)
        return None

    encoded_text = text_dict["encoded_text"]
    text_token_mask = text_dict["text_token_mask"]
    position_ids = text_dict["position_ids"]
    text_self_attention_masks = text_dict["text_self_attention_masks"]

    return (encoded_text,
            text_token_mask,
            position_ids,
            text_self_attention_masks)

def TRT_load_data(image, image_size, text_prompt):
    image_resized = TRT_resize_image(image, image_size)
    (encoded_text,
     text_token_mask,
     position_ids,
     text_self_attention_masks) = TRT_load_text_embedding(text_prompt)

    inputs_dict = {
        "img": image_resized[None].numpy().astype(np.float32),
        "encoded_text": encoded_text.numpy().astype(np.float32),
        "text_token_mask": text_token_mask.numpy().astype(np.bool_),
        "position_ids": position_ids.numpy().astype(np.int32),
        "text_self_attention_masks": text_self_attention_masks.numpy().astype(np.bool_),
    }
    return inputs_dict

## Post-processing -----------------------------
def load_predict_phrase(text_prompt):
    cache_path = get_text_dict_cache_path(preprocess_caption(caption=text_prompt),predict=True)
    if os.path.exists(cache_path):
        # Load to gpu for faster post-processing
        pred_phrase = torch.load(cache_path, map_location='cuda')
    else:
        print('Predict phrase cache not found! Text prompt: ',text_prompt,
              'Cache path: ',cache_path)
        return None
    return pred_phrase

def cal_predict_phrase(logits,
                       text_threshold,
                       text_prompt,
                       remove_combined = False,
                       cache_path = "./text_embed_cache"):
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       "./weights/groundingdino_swint_ogc.pth")
    model.to('cuda')
    tokenizer = model.tokenizer
    tokenized = tokenizer(text_prompt)
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.',
                                                                                                                   ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
    torch.save(phrases, cache_path)
    return phrases

def cal_predict_results(model_outputs,
                        box_threshold,
                        text_threshold,
                        text_prompt,):
    pred_phrase = load_predict_phrase(text_prompt)
    pred_logits = torch.from_numpy(model_outputs[0].reshape(1, 900, 256)).cpu().sigmoid()[0]
    pred_boxes = torch.from_numpy(model_outputs[1].reshape(1, 900, 4)).cpu()[0]
    # pred_logits.shape = (1, nq, 256)
    # pred_boxes.shape = (1, nq, 4)

    mask = pred_logits.max(dim=1)[0] > box_threshold
    logits = pred_logits[mask]  # logits.shape = (1, n, 256)
    boxes = pred_boxes[mask]  # boxes.shape = (1, n, 4)
    if pred_phrase is None:
        # Calculate predict phrases if not preloaded
        print('Predict phrase cache for calculation not found!',
              'Text prompt: ',text_prompt,
              ' Loading model to calculate!')
        pred_phrase = cal_predict_phrase(logits = logits,
                                         text_threshold = text_threshold,
                                         text_prompt = text_prompt)
    else:
        # Load predict phrases for labeling
        caption = preprocess_caption(caption=text_prompt)
        phrases = pred_phrase if pred_phrase else [caption]  # Use TEXT_PROMPT as fallback labels
        if len(phrases) < len(boxes):
            phrases.extend([caption] * (len(boxes) - len(phrases)))  # Fill missing labels
        elif len(phrases) > len(boxes):
            phrases = phrases[:len(boxes)]  # Trim extra labels
        pred_phrase = phrases

    # Calculate
    return boxes, logits.max(dim=1)[0], pred_phrase