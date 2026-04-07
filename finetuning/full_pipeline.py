"""
full_pipeline.py
----------------
Does everything in one shot:
  1. Extract LoRA adapter from zip
  2. Merge adapter into Qwen2.5-3B-Instruct base model
  3. Download llama.cpp GGUF converter (pure Python, no binary needed)
  4. Convert merged model → GGUF (f16)

Run:
    pip install transformers peft torch accelerate gguf
    python finetuning/full_pipeline.py

Then register with Ollama:
    ollama create qwen2.5-banking:3b -f finetuning/Modelfile
    ollama run qwen2.5-banking:3b "How do I open a savings account?"
"""

import subprocess
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ZIP_PATH    = SCRIPT_DIR / "finetuning_results" / "download_bestcheckpoin.zip"
ADAPTER_DIR = SCRIPT_DIR / "finetuning_results" / "lora_adapter"
MERGED_DIR  = SCRIPT_DIR / "finetuning_results" / "merged_model"
GGUF_OUT    = SCRIPT_DIR / "finetuning_results" / "merged_model_f16.gguf"
CONVERTER   = SCRIPT_DIR / "finetuning_results" / "convert_hf_to_gguf.py"

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"

CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggerganov/llama.cpp"
    "/master/convert_hf_to_gguf.py"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def banner(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ── Step 1 — Extract LoRA adapter ────────────────────────────────────────────
banner("Step 1 — Extracting LoRA adapter")

if ADAPTER_DIR.exists():
    shutil.rmtree(ADAPTER_DIR)
ADAPTER_DIR.mkdir(parents=True)

with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall(ADAPTER_DIR)

print(f"  Files: {[p.name for p in ADAPTER_DIR.iterdir()]}")
print(f"  -> {ADAPTER_DIR}")


# ── Step 2 — Merge LoRA into base model ──────────────────────────────────────
banner("Step 2 — Merging LoRA into base model (downloads ~6 GB first run)")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
print("  Tokenizer loaded.")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
print("  Base model loaded.")

model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
print("  Adapter attached. Merging ...")
model = model.merge_and_unload()
print("  Merged.")

MERGED_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_DIR))
print(f"  -> Saved to {MERGED_DIR}")

# Free memory before conversion
del model
import gc; gc.collect()


# ── Step 3 — Download llama.cpp GGUF converter ───────────────────────────────
banner("Step 3 — Downloading GGUF converter from llama.cpp")

if not CONVERTER.exists():
    print(f"  Fetching {CONVERTER_URL} ...")
    urllib.request.urlretrieve(CONVERTER_URL, CONVERTER)
    print(f"  -> Saved to {CONVERTER}")
else:
    print(f"  Already present: {CONVERTER}")


# ── Step 4 — Convert to GGUF ─────────────────────────────────────────────────
banner("Step 4 — Converting to GGUF (f16)")

result = subprocess.run(
    [
        sys.executable, str(CONVERTER),
        str(MERGED_DIR),
        "--outfile", str(GGUF_OUT),
        "--outtype", "f16",
    ],
    capture_output=False,
)

if result.returncode != 0:
    print("\n[ERROR] GGUF conversion failed.")
    print("  Make sure 'gguf' is installed:  pip install gguf")
    sys.exit(1)

print(f"\n  -> GGUF saved: {GGUF_OUT}")
size_gb = GGUF_OUT.stat().st_size / 1e9
print(f"  -> Size: {size_gb:.1f} GB")


# ── Done ─────────────────────────────────────────────────────────────────────
banner("All done!")
print("Next steps:")
print()
print("  1. Register with Ollama:")
print("       ollama create qwen2.5-banking:3b -f finetuning/Modelfile")
print()
print("  2. Test:")
print("       ollama run qwen2.5-banking:3b \"How do I open a savings account?\"")
print()
print("  3. Run the RAG app with the new model:")
print("       set OLLAMA_CHAT_MODEL=qwen2.5-banking:3b")
print("       uv run python main.py")
