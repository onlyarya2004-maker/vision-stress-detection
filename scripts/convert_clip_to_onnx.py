"""
One-time preparation of the CLIP "hint" model for the LOCAL-ONLY context
suggestion (code/context_hints.py).

Run with the TRAINING venv (the only one with PyTorch + transformers):

    .venv-train\\Scripts\\python.exe scripts\\convert_clip_to_onnx.py

Downloads openai/clip-vit-base-patch32 (MIT licence, ~600 MB, Hugging Face
cache) and writes, into models/context/clip/ (gitignored; copied into the Docker
image by the existing `COPY models` line):
    clip_image_encoder.onnx   image -> unnormalised 512-d embedding
    clip_text_embeds.npz      normalised text embeddings of every sentence in
                              code/context_hint_prompts.py, its group labels,
                              and CLIP's logit scale
    parity_reference.npz      inputs + torch embeddings for the parity test

The text side runs here, once, so the app needs only the image encoder.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from context_hint_prompts import HINT_PROMPTS  # noqa: E402

OUT = ROOT / 'models' / 'context' / 'clip'
NAME = 'openai/clip-vit-base-patch32'


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(NAME).eval()
    tok = CLIPTokenizer.from_pretrained(NAME)

    prompts, groups = [], []
    for g, sentences in HINT_PROMPTS.items():
        prompts += sentences
        groups += [g] * len(sentences)
    with torch.no_grad():
        t = model.get_text_features(**tok(prompts, padding=True, return_tensors='pt'))
        t = (t / t.norm(dim=-1, keepdim=True)).numpy()
    np.savez_compressed(OUT / 'clip_text_embeds.npz', text=t, prompts=np.array(prompts),
                        groups=np.array(groups), logit_scale=float(model.logit_scale.exp()))

    enc = ImageEncoder(model).eval()
    torch.onnx.export(enc, torch.zeros(1, 3, 224, 224), str(OUT / 'clip_image_encoder.onnx'),
                      input_names=['pixel_values'], output_names=['embeds'],
                      opset_version=14, dynamo=False)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(2, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = enc(torch.from_numpy(x)).numpy()
    np.savez_compressed(OUT / 'parity_reference.npz', inputs=x, embeds=ref)
    size = (OUT / 'clip_image_encoder.onnx').stat().st_size / 1e6
    print(f"wrote {len(prompts)} text embeddings and clip_image_encoder.onnx ({size:.0f} MB) to {OUT}")


if __name__ == '__main__':
    main()
