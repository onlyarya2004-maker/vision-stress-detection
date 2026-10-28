# models/context/ — models for the "Auto-detect" context suggestion

Two small pretrained models, run on the server's/laptop's CPU through OpenCV DNN (no new
dependency, no external API). Their converted files are **gitignored** (the CLIP encoder is 352 MB,
over GitHub's 100 MB file limit) but they sit under `models/`, which the Dockerfile already copies, so
an image built from a checkout that has them contains them. Without them the app still works; the
Auto-detect button then says context detection is unavailable. Set `STRESS_AUTO_CONTEXT=0` to hide
the button. Raw downloads and the small public test-photo set live in the gitignored `local_models/`.

## places365/ — scene model (room types)

| File | Size | SHA-256 | Origin |
|---|---|---|---|
| `local_models/places365/resnet18_places365.pth.tar` (raw) | 45,506,139 B | `2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588` | http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar |
| `models/context/places365/categories_places365.txt` | 6,833 B | `2affba635eb657e7ca95f4e6cc69bd9fac29ef4c32aeb83cafdfcd06ec6a1ea6` | https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt |
| `models/context/places365/resnet18_places365.onnx` | ≈45.4 MB | derived | `scripts/convert_places365_to_onnx.py` |
| `models/context/places365/parity_reference.npz` | 2 MB | derived (not needed in the image) | the same script (torch outputs used by `test_context_detector.py`) |

**Model:** Places365 ResNet18 (365 scene categories), by B. Zhou et al., MIT CSAIL, from
https://github.com/CSAILVision/places365.

**Licence:** the repository code is MIT, and the pretrained models are released under
Creative Commons Attribution (CC BY). Credit to cite: B. Zhou, A. Lapedriza, A. Khosla, A. Oliva,
A. Torralba, "Places: A 10 million Image Database for Scene Recognition", IEEE TPAMI, 2017.

**Runtime:** OpenCV's DNN module (`cv2.dnn.readNetFromONNX`), which is already in
`requirements.txt`. No new runtime package is needed. Parity with PyTorch was checked: maximum
absolute logit difference is 8.6e-6 and the top-1 class is identical. Speed is about 19 ms per
frame on the development PC.

### Re-create from scratch
```bash
mkdir -p local_models/places365 && cd local_models/places365
curl -LO http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
curl -LO https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt
cd ../..
# The conversion needs torch (training venv) plus the `onnx` package. onnx==1.17.0 works with
# that venv's protobuf 4.25 pin; newer onnx needs protobuf>=6 and breaks the mediapipe pin.
.venv-train/Scripts/python.exe -m pip install "onnx==1.17.0" "protobuf==4.25.9"
.venv-train/Scripts/python.exe scripts/convert_places365_to_onnx.py
.venv-train/Scripts/python.exe -m pip uninstall -y onnx     # leave the venv as it was
```

## clip/ — CLIP "hints" (people and objects)
| File | Size | Origin |
|---|---|---|
| `clip_image_encoder.onnx` | 352 MB | `scripts/convert_clip_to_onnx.py` from `openai/clip-vit-base-patch32` (Hugging Face, MIT licence; Radford et al., 2021) |
| `clip_text_embeds.npz` | 51 KB | the same script: embeddings of every sentence in `code/context_hint_prompts.py` |
| `parity_reference.npz` | 1 MB | torch outputs for `test_context_detector.py` |

Zero-shot: each frame is compared with sentences such as "a student writing in a notebook", "a doctor
wearing a white coat", "a person sitting inside a car". The probability falling on each group
(Studying / Hospital / Vehicle / Other) is blended with the scene model's evidence. Runs through OpenCV DNN
(max diff vs PyTorch 4e-6, ~70 ms/frame); no new app dependency. Needs `transformers` and `onnx` only for the
one-time conversion, in the training venv (restore it afterwards, as for Places365). If the files are absent
the feature silently falls back to the scene model alone. Editing a sentence means re-running the script.
