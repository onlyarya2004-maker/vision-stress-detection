"""
One-time conversion of the official Places365 ResNet18 weights to ONNX, for
the LOCAL-ONLY "Auto-detect context" feature (code/context_detector.py).

Run with the TRAINING venv - it is the only environment with PyTorch, and the
live app never imports torch:

    .venv-train\\Scripts\\python.exe scripts\\convert_places365_to_onnx.py

Input  (download first, see local_models/places365/SOURCES.md):
    local_models/places365/resnet18_places365.pth.tar
Output (models/context/places365/, gitignored but under models/, which the Docker
image copies):
    models/context/places365/resnet18_places365.onnx
    models/context/places365/parity_reference.npz   (inputs + torch logits, so
        tests/test_context_detector.py can prove the OpenCV runtime matches)

The checkpoint is read with torch.load(weights_only=True): tensors only, no
arbitrary pickled code from the download is executed.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'local_models' / 'places365'          # downloaded weights (gitignored)
DIR = ROOT / 'models' / 'context' / 'places365'    # converted files the app (and image) read
SRC = RAW / 'resnet18_places365.pth.tar'
DST = DIR / 'resnet18_places365.onnx'
REF = DIR / 'parity_reference.npz'


def main():
    ckpt = torch.load(SRC, map_location='cpu', weights_only=True)
    state = {k.replace('module.', '', 1): v for k, v in ckpt['state_dict'].items()}
    model = torchvision.models.resnet18(num_classes=365)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, str(DST), input_names=['input'],
                      output_names=['logits'], opset_version=13, dynamo=False)

    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(4, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
    np.savez_compressed(REF, inputs=x, logits=logits)
    print(f"wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB) and {REF.name}")


if __name__ == '__main__':
    main()
