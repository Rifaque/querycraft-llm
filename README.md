# QueryCraft-LLM — Quick start (Windows / RTX 4060)

Welcome! This README is written for friends on Windows machines (RTX 4060). It walks through cloning the repo, creating a Python virtual environment, installing GPU-enabled PyTorch and the other Python packages, generating a JSONL dataset, and a minimal path to start training a small / fine-tuning workflow.

> ⚠️ Quick summary:
>
> * **Recommended:** Use WSL2 (Ubuntu) on Windows for the smoothest CUDA / bitsandbytes experience. Windows-native can work for many libraries, but some LLM training/quantization libraries are supported only on Linux/WSL.

---

## Prerequisites

1. Windows 10/11 with WSL2 available (recommended) OR Windows PowerShell / CMD.

   * If you plan to train on the RTX 4060 with CUDA support, **WSL2 + Ubuntu** + NVIDIA CUDA-enabled WSL driver is the most reliable setup.
2. Python 3.10 or 3.11 (3.10 is a very stable choice for many DL libraries).
3. Git
4. (Optional but recommended) 20–50 GB free disk for model weights and checkpoints.

---

## 1) Clone the repo

```bash
git clone https://github.com/Rifaque/querycraft-llm.git
cd querycraft-llm
```

---

## 2) Create & activate a virtual environment

### PowerShell (Windows native)

```powershell
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# If running in cmd:
# .\.venv\Scripts\activate.bat
python -m pip install -U pip setuptools wheel
```

### Bash (recommended inside WSL / Ubuntu)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

---

## 3) Install PyTorch (CUDA) and the rest

**Important**: PyTorch wheels are CUDA-specific. For an RTX 4060 you will most likely want a CUDA 12.x wheel (PyTorch provides CUDA 12.x wheels — choose the CUDA version that matches your driver/WSL setup).

### PowerShell (Windows) - copy/paste script

```powershell
# 1) Make sure venv is activated: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel

# 2) Install PyTorch with CUDA 12.x wheels (change cu121 if you need another CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) Install bitsandbytes and other packages (bitsandbytes may be unreliable on Windows; see note below)
pip install bitsandbytes || Write-Host "bitsandbytes install failed — see README for WSL recommendation"

# 4) Install the rest
pip install -r requirements.txt
```

### Bash (WSL / Ubuntu) - copy/paste script

```bash
# 1) Activate venv: source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 2) Install PyTorch with CUDA 12.x wheels (adjust cu121 if required for your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) Install bitsandbytes (Linux/WSL is supported and recommended)
pip install bitsandbytes

# 4) Install the rest
pip install -r requirements.txt
```

**Notes & troubleshooting**

* `bitsandbytes` historically has had Linux-first support and installing it on Windows can be fragile; using WSL2/Ubuntu avoids many issues. If `bitsandbytes` fails on Windows, switch to WSL2 or skip it (you can still train smaller models without `bitsandbytes`).
* If `pip install torch ...` fails, double-check your CUDA driver and WSL NVIDIA driver; PyTorch wheels assume the system driver supports the required CUDA stack.

---

## 4) `requirements.txt`

A recommended `requirements.txt` to put in the repo root:

```
# NOTE: torch is installed separately because of CUDA wheel selection.
# NOTE: bitsandbytes may require Linux/WSL for GPU support; install separately as above.

accelerate>=0.21.0
transformers>=4.35.0
datasets>=2.14.0
peft
safetensors
tokenizers
huggingface-hub
sentencepiece
tqdm
numpy
pandas
rich
fire
wandb
```

You can tweak versions to match your project; `accelerate` helps with distributed & mixed-precision training.

---

## 5) Generate the JSONL dataset

This repo expects training data as JSONL (one JSON object per line). Run:

```bash
python dataset_generator.py
```

---

## 6) Training workflow (example)

1. Prepare `dataset.jsonl` (see step 5).

2. Train/Create the model:
```bash
# Example: run from inside the venv
python train.py
```

3. Test the model:
```bash
# Example: run from inside the venv
python test1.py
```

---

## 7) Tips for RTX 4060 owners

* The RTX 4060 is a capable card; for large models you will need quantization (e.g., 8-bit/4-bit) and/or gradient checkpointing to fit models. `bitsandbytes` + `peft` (LoRA / QLoRA) helps a lot — but `bitsandbytes` works best on Linux/WSL.
* If you hit CUDA/bitsandbytes errors on Windows, move to WSL2 (Ubuntu). Most LLM fine-tuning guides recommend WSL or Linux for fewer headaches.

---

## 8) If things fail (quick checklist)

* Is the venv activated? `python -m pip list`
* Is your NVIDIA driver (Windows host) up-to-date and WSL driver installed?
* If `bitsandbytes` fails on Windows, try in WSL2/Ubuntu.
* When in doubt, test CUDA visibility with:

```python
python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

---

## 9) Final notes

* This README gives a portable, reproducible start: clone, create venv, install CUDA-enabled torch, install extras, generate JSONL, then train.

Happy training! 🚀
