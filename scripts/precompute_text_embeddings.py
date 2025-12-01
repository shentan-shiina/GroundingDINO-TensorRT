#!/usr/bin/env python3
"""
Precompute and save text embedding caches for GroundingDINO prompts.

Usage:
  python scripts/precompute_text_embeddings.py --csv prompts.csv \
      --model-config groundingdino/config/GroundingDINO_SwinT_OGC.py \
      --checkpoint weights/groundingdino_swint_ogc.pth \
      --device cpu --max-len 128

The script reads the first column of the CSV as text prompts, computes the
text_dict (encoded_text, text_token_mask, position_ids, text_self_attention_masks)
via the model's tokenizer + bert encoder + feat_map, then pads/truncates all
saved caches to the same sequence length (`--max-len` or the max seen).

It saves each cache using `get_text_dict_cache_path(prompt)` so the rest of the
repo (TensorRT utilities) can load them.
"""
import argparse
import csv
import os
from typing import List

import torch

from groundingdino.util.inference import load_model, preprocess_caption
from groundingdino.util.utils import get_text_dict_cache_path
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

def compute_text_dict_for_prompt(model, prompt: str, device: str = "cpu") -> dict:
    caption = preprocess_caption(prompt)
    tokenizer = model.tokenizer

    # Tokenize (single sample)
    tokenized = tokenizer(caption, padding=False, return_tensors="pt")

    # Generate attention masks / position ids consistent with model
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, model.specical_tokens, model.tokenizer
    )

    # Move tensors to device
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    text_self_attention_masks = text_self_attention_masks.to(device)
    position_ids = position_ids.to(device)

    # Prepare tokenized_for_encoder as done in model.forward
    if model.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
    else:
        tokenized_for_encoder = tokenized

    # Run through BERT warper and feat_map to get encoded_text
    with torch.no_grad():
        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])  # (1, seq_len, d_model)

    # Prepare masks and move to cpu
    text_token_mask = tokenized["attention_mask"].bool().cpu()  # (1, seq_len)
    encoded_text = encoded_text.cpu()
    position_ids = position_ids.cpu()
    text_self_attention_masks = text_self_attention_masks.cpu()

    text_dict = {
        "encoded_text": encoded_text,  # torch.Tensor (1, seq_len, d_model)
        "text_token_mask": text_token_mask,  # torch.Tensor (1, seq_len)
        "position_ids": position_ids,  # torch.Tensor (1, seq_len)
        "text_self_attention_masks": text_self_attention_masks,  # torch.Tensor (1, seq_len, seq_len)
    }
    return text_dict


def pad_text_dict(text_dict: dict, target_len: int) -> dict:
    """Pad or truncate tensors in text_dict to target_len along the sequence axis.

    Assumes tensors shapes:
      encoded_text: (1, L, D)
      text_token_mask: (1, L)
      position_ids: (1, L)
      text_self_attention_masks: (1, L, L)
    """
    import torch

    enc = text_dict["encoded_text"]
    tok = text_dict["text_token_mask"]
    pos = text_dict["position_ids"]
    attn = text_dict["text_self_attention_masks"]

    cur_len = enc.shape[1]
    D = enc.shape[2]
    if cur_len == target_len:
        return text_dict

    # Truncate or pad encoded_text
    if cur_len > target_len:
        enc2 = enc[:, :target_len, :]
        tok2 = tok[:, :target_len]
        pos2 = pos[:, :target_len]
        attn2 = attn[:, :target_len, :target_len]
    else:
        # pad
        pad = (0, 0, 0, target_len - cur_len)  # for 3D: pad last two dims
        enc2 = torch.nn.functional.pad(enc, pad)
        # for 2D masks
        tok2 = torch.nn.functional.pad(tok, (0, target_len - cur_len), value=False)
        pos2 = torch.nn.functional.pad(pos, (0, target_len - cur_len), value=0)
        # attn: pad last two dims
        attn2 = torch.nn.functional.pad(attn, (0, target_len - cur_len, 0, target_len - cur_len), value=False)

    return {
        "encoded_text": enc2,
        "text_token_mask": tok2,
        "position_ids": pos2,
        "text_self_attention_masks": attn2,
    }


def main(csv_path: str, device: str = "cpu", max_len: int = None):
    # Load model
    print("Loading model...")
    model = load_model("./groundingdino/config/GroundingDINO_SwinT_OGC.py", "./weights/groundingdino_swint_ogc.pth")
    model.eval()

    prompts: List[str] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            prompts.append(row[0].strip())

    print(f"Found {len(prompts)} prompts")

    # Compute text_dicts
    results = []
    for p in prompts:
        try:
            td = compute_text_dict_for_prompt(model, p, device=device)
            results.append((p, td))
        except Exception as e:
            print(f"Failed to compute embedding for '{p}': {e}")

    # Determine target length
    lengths = [td[1]["encoded_text"].shape[1] for td in results]
    observed_max = max(lengths) if lengths else 0
    if max_len is None:
        # Use observed max but cap to model.max_text_len if present
        cap = getattr(model, 'max_text_len', None)
        if cap is not None:
            target_len = min(observed_max, cap)
        else:
            target_len = observed_max
    else:
        target_len = int(max_len)

    print(f"Using target text length = {target_len}")

    # Save caches
    import torch
    for prompt, td in results:
        td_padded = pad_text_dict(td, target_len)
        cache_path = get_text_dict_cache_path(preprocess_caption(prompt))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(td_padded, cache_path)
        # save a simple predict cache (fallback phrases list)
        pred_cache = get_text_dict_cache_path(preprocess_caption(prompt), predict=True)
        os.makedirs(os.path.dirname(pred_cache), exist_ok=True)
        # save simple label list containing the caption as fallback
        torch.save([preprocess_caption(prompt)], pred_cache)
        print(f"Saved caches for prompt: {prompt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=False, default='./scripts/text_prompt.csv',help='CSV file with prompts (first column)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-len', type=int, default=None, help='Force fixed length; otherwise use observed max (capped by model)')
    args = parser.parse_args()

    main(args.csv, device=args.device, max_len=15)
