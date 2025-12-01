import torch
import numpy as np
from groundingdino.util.inference import load_model, load_image
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.utils import get_text_dict_cache_path
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
import os
from groundingdino.util.misc import nested_tensor_from_tensor_list
import onnxruntime as ort

# Load the GroundingDINO model
def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

# model = load_model(model_config_path="./groundingdino/config/GroundingDINO_SwinT_OGC.py", model_checkpoint_path="./weights/groundingdino_swint_ogc.pth", cpu_only=True)

model = load_model(model_config_path="./groundingdino/config/GroundingDINO_SwinB_cfg.py", model_checkpoint_path="./weights/groundingdino_swinb_cogcoor.pth", cpu_only=True)

TEXT_PROMPT = "white pen ."
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

cache_path = get_text_dict_cache_path(TEXT_PROMPT)
if os.path.exists(cache_path):
    text_dict = torch.load(cache_path, map_location='cpu')

# Dummy image input (batch_size=1, 3 channels, H=640, W=640)
img = torch.randn(1, 3, 400, 400) # Adjust size as needed

encoded_text = text_dict["encoded_text"]
text_token_mask = text_dict["text_token_mask"]
position_ids = text_dict["position_ids"]
text_self_attention_masks = text_dict["text_self_attention_masks"]

dynamic_axes = {
    "img": {0: "batch_size", 2: "height", 3: "width"},
    "encoded_text": {0: "batch_size", 1: "seq_len"},
    "text_token_mask": {0: "batch_size", 1: "seq_len"},
    "position_ids": {0: "batch_size", 1: "seq_len"},
    "text_self_attention_masks": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
    "logits": {0: "batch_size"},
    "boxes": {0: "batch_size"}
}

onnx_path = "./.asset/groundingdino_v4_400_2_30_b.onnx"
# export onnx model
torch.onnx.export(
    model,
    f=onnx_path,
    args=(img, encoded_text, text_token_mask, position_ids, text_self_attention_masks),
    # input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids","text_token_mask"],
    input_names=["img", "encoded_text", "text_token_mask", "position_ids", "text_self_attention_masks"],
    output_names=["logits", "boxes"],
    dynamic_axes=dynamic_axes,
    # do_constant_folding=True,
    opset_version=16)

print(f"Exported ONNX model to {onnx_path}")

session = ort.InferenceSession(onnx_path)

output_names = ["logits", "boxes"]

inputs = {
    'img': img.cpu().numpy(), # 1*3*N*N
    'encoded_text': encoded_text.cpu().numpy(), # 1*n*256
    'text_token_mask': text_token_mask.cpu().numpy(), # 1*n
    'position_ids': position_ids.cpu().numpy(), # 1*n
    'text_self_attention_masks': text_self_attention_masks.cpu().numpy(), # 1*n*n
}

outputs = session.run(output_names, inputs)
