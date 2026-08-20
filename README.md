# AI-Powered Waste Segregation Model

A smart waste-management system that uses **computer vision** and **large language models** to automatically identify waste items from a camera image and recommend the correct colour-coded disposal bin.

---

## Problem Statement

Manual waste segregation is inconsistent, labour-intensive, and error-prone. Campus and household bins are frequently contaminated because users are unsure which items belong in which stream. This project proposes an **automated, image-based waste classification pipeline** that:

1. Captures or receives a photo of a waste item.
2. Classifies it into one of **seven waste categories** using a deep-learning model.
3. Maps the prediction to a colour-coded bin (Green / Blue / Red) with clear disposal instructions.
4. (Optional) Passes the structured prediction to a **local LLM** (Llama 3.2 via Ollama) that generates a natural-language disposal guide.

The goal is to reduce contamination rates and make waste segregation effortless for any user.

---

## System Architecture

```
┌──────────────┐    ┌──────────────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Camera /   │───▶│  CV Classifier            │───▶│  Bin Mapper     │───▶│  LLM Guidance    │
│   Upload     │    │  (MobileNetV3-Small)      │    │  (Rule-based)   │    │  (Ollama / LLM)  │
│   Image      │    │  7-class softmax          │    │  GREEN / BLUE / │    │  Natural-language│
└──────────────┘    │  + confidence score       │    │  RED assignment │    │  disposal tips   │
                    └──────────────────────────┘    └─────────────────┘    └──────────────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │  Gradio Web App  │
                                                   │  localhost:7860   │
                                                   └──────────────────┘
```

### Pipeline in detail

| Stage | Component | Description |
|-------|-----------|-------------|
| 1. Input | `app.py` (Gradio) | User uploads or captures an image via webcam / file upload. |
| 2. Classification | `src/predict.py` + `src/model.py` | Image is resized to 224x224, normalised, and passed through a fine-tuned MobileNetV3-Small. The 7-class softmax output provides a label and confidence score. |
| 3. Bin mapping | `src/recommend.py` | The predicted class is mapped to a colour-coded bin (GREEN, BLUE, or RED) with specific disposal instructions. |
| 4. LLM layer | `src/llm.py` | When Ollama is running, the structured CV output is sent to Llama 3.2 3B; the LLM returns a concise, user-friendly disposal guide. |
| 5. Display | `app.py` | Results are rendered as a colour-coded card with confidence bar, detected class, bin colour, and instructions. |

---

## Waste Classes and Bin Mapping

| Class | Bin | Colour | Stream |
|-------|-----|--------|--------|
| **organic** | GREEN | ![#4CAF50](https://via.placeholder.com/12/4CAF50/4CAF50.png) #4CAF50 | Biodegradable / Organic waste |
| **cardboard** | BLUE | ![#2196F3](https://via.placeholder.com/12/2196F3/2196F3.png) #2196F3 | Recyclable / Dry waste |
| **paper** | BLUE | ![#2196F3](https://via.placeholder.com/12/2196F3/2196F3.png) #2196F3 | Recyclable / Dry waste |
| **plastic** | BLUE | ![#2196F3](https://via.placeholder.com/12/2196F3/2196F3.png) #2196F3 | Recyclable / Dry waste |
| **metal** | BLUE | ![#2196F3](https://via.placeholder.com/12/2196F3/2196F3.png) #2196F3 | Recyclable / Dry waste |
| **glass** | RED | ![#EF4444](https://via.placeholder.com/12/EF4444/EF4444.png) #EF4444 | Non-recyclable / Special waste |
| **trash** | RED | ![#EF4444](https://via.placeholder.com/12/EF4444/EF4444.png) #EF4444 | Non-recyclable / Residual |

---

## Dataset

Two public datasets were combined:

| Dataset | Source | Classes | Notes |
|---------|--------|---------|-------|
| **TrashNet** | [Garythung/trashnet](https://github.com/garythung/trashnet) | cardboard, glass, metal, paper, plastic, trash | 1,769 images (224x224, resized) |
| **AlphaTrash (organic subset)** | [Patipol-BKK/alphatrash-dataset](https://github.com/Patipol-BKK/alphatrash-dataset) | organic | 567 food / organic waste images |

**Total:** 2,336 images across 7 classes, split 70 / 15 / 15 (train / val / test).

---

## Model Architecture

| Property | Value |
|----------|-------|
| **Backbone** | MobileNetV3-Small (ImageNet-pretrained) |
| **Fine-tuning** | Transfer learning — final classifier head replaced with `Linear(576, 7)` |
| **Input size** | 224 x 224 x 3 (RGB), normalised with ImageNet mean/std |
| **Output** | 7-class softmax (cardboard, glass, metal, organic, paper, plastic, trash) |
| **Framework** | PyTorch + torchvision |
| **Checkpoint** | `models/best_model.pth` (~6 MB) |

### Training configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 15 |
| Scheduler | None (fixed LR) |
| Augmentation | RandomHorizontalFlip, RandomRotation(15), ColorJitter, RandomResizedCrop |
| Device | CPU |

### Validation results

| Metric | Value |
|--------|-------|
| **Best val accuracy** | **92.5 %** (epoch 9) |
| Train accuracy (at best val) | 98.4 % |
| Inference speed (CPU) | ~1 image / 0.9 s |

---

## Project Structure

```
.
├── app.py                  Gradio web demo (localhost)
├── requirements.txt        Python dependencies
├── run_demo.bat            One-click launcher (Windows)
├── .env.example            Environment config template
│
├── src/
│   ├── config.py           Paths, class names, hyperparameters
│   ├── dataset.py          ImageFolder loaders + augmentation
│   ├── model.py            MobileNetV3 / EfficientNet / ResNet factory
│   ├── train.py            Training entry point (CLI)
│   ├── predict.py          Checkpoint loading + image inference
│   ├── recommend.py        Class → bin colour / stream / instructions
│   └── llm.py              Ollama LLM integration (local)
│
├── scripts/
│   ├── download_trashnet.py    Downloads TrashNet dataset (~43 MB)
│   ├── download_organic.py     Downloads organic waste images from AlphaTrash
│   └── prepare_data.py         70/15/15 split into data/{train,val,test}
│
├── models/
│   └── best_model.pth      Trained checkpoint (~6 MB)
│
└── data/
    ├── dataset-resized/    Raw images by class
    ├── train/              70 % training split
    ├── val/                15 % validation split
    └── test/               15 % test split
```

---

## Quick Start (Windows / PowerShell)

```powershell
# 1. Create virtual environment
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip

# 2. Install dependencies (CPU-only torch)
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -m pip install -r requirements.txt

# 3. Download datasets
py scripts\download_trashnet.py
py scripts\download_organic.py
py scripts\prepare_data.py

# 4. Train the model (~15 min on CPU)
py -m src.train --arch mobilenet_v3_small --epochs 15 --batch-size 32 --device cpu

# 5. Launch the web app
py app.py --checkpoint models\best_model.pth --port 7860
# Open http://127.0.0.1:7860
```

Or run everything at once:

```powershell
.\run_demo.bat
```

---

## CLI Usage

```powershell
# Classify a single image
py -m src.predict --checkpoint models\best_model.pth --image path\to\photo.jpg

# Classify a folder (JSON output)
py -m src.predict --checkpoint models\best_model.pth --dir samples --json

# Training with different architecture
py -m src.train --arch resnet18 --epochs 25 --lr 1e-3
```

Available architectures: `mobilenet_v3_small`, `resnet18`, `resnet50`, `efficientnet_b0`.

---

## LLM Layer (Local, No API Key)

The app uses **Ollama** to run Llama 3.2 3B locally for waste disposal guidance.

1. Install Ollama from https://ollama.com/download
2. Pull the model:
   ```
   ollama pull llama3.2:3b
   ```
3. Start the app — the LLM layer activates automatically when Ollama is running.

The LLM only receives **text** (the CV model's structured prediction), never images.

---

## Demo Video

See `demo_video.mp4` in the repository root for a screen-recorded walkthrough of the model classifying sample waste items across all 7 categories.

---

## Dependencies

- Python 3.12+
- PyTorch (CPU or CUDA)
- torchvision
- Gradio
- openai (Ollama uses OpenAI-compatible API)
- python-dotenv
- tqdm
- Pillow

---

## License

This project was developed as part of coursework (CIA 2 — 2582422). For academic use only.
