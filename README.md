# AI-Powered Waste Segregation Model

Computer-vision waste classifier + LLM guidance for a smart campus waste-management
system. A camera image is classified by a PyTorch model, then an LLM (OpenAI) turns the
structured prediction into a clear, color-coded disposal recommendation.

Pipeline: **Capture -> CV classifier (what is it?) -> LLM (what should I do?) -> Bin recommendation**

## Project layout

```
src/
  config.py      paths, class names, hyperparameters
  dataset.py     ImageFolder loaders + augmentation transforms
  model.py       MobileNet / EfficientNet / ResNet transfer-learning factory
  train.py       training entry point (CLI)
  predict.py     checkpoint loading + image inference (CLI)
  recommend.py   class -> bin color / stream / instructions
  llm.py         OpenAI integration (structured prompt from CV output)
scripts/
  download_trashnet.py   downloads the TrashNet dataset (~43 MB)
  prepare_data.py        70/15/15 split into data/{train,val,test}
app.py           Gradio web demo (localhost)
```

## Quick start (Windows / localhost)

Open **PowerShell** in this folder and run:

```powershell
# 1) Install dependencies (virtual environment)
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# CPU-only torch is smaller/faster to install (optional):
# py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Get data
py scripts\download_trashnet.py
py scripts\prepare_data.py

# 3) Train the classifier (CPU: ~20-40 min for resnet18/25 epochs)
py -m src.train --arch resnet18 --epochs 25

# 4) Launch the web app -> open http://127.0.0.1:7860
py app.py --checkpoint models\best_model.pth
```

Or run everything with one script:

```powershell
.\run_demo.bat
```

### Add the LLM layer

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY=sk-...`
   (the app auto-loads `.env`).
2. Restart `py app.py ...`.

Without a key the app still works locally using rule-based bin guidance.

## CLI usage

```powershell
# Classify one image
py -m src.predict --checkpoint models\best_model.pth --image path\to\photo.jpg

# Classify a folder, machine-readable
py -m src.predict --checkpoint models\best_model.pth --dir samples --json
```

## Training options

```powershell
py -m src.train --arch efficientnet_b0 --epochs 40 --batch-size 64 --lr 1e-3
```

Available architectures: `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v3_small`.

## Classes and bin mapping

| Class       | Bin   | Stream                        |
|-------------|-------|-------------------------------|
| organic     | GREEN | Biodegradable / Organic waste |
| cardboard   | BLUE  | Recyclable / Dry              |
| paper       | BLUE  | Recyclable / Dry              |
| plastic     | BLUE  | Recyclable / Dry              |
| metal       | BLUE  | Recyclable / Dry              |
| glass       | RED   | Non-recyclable / Special      |
| trash       | RED   | Non-recyclable / Residual     |

The `organic` bin is pre-defined. To train it, add food/organic-waste images in
`data/dataset-resized/organic/`, run `py scripts\prepare_data.py` (it auto-detects
class folders), then retrain.
