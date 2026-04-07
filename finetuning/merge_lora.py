"""
merge_lora.py
-------------
Extracts the best checkpoint LoRA adapter from zip, merges it with the
Qwen/Qwen2.5-3B-Instruct base model, and saves a standard HuggingFace model.

Run from the project root:
    python finetuning/merge_lora.py

Requirements:
    pip install transformers peft torch accelerate
"""

import zipfile
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
ZIP_PATH     = SCRIPT_DIR / "finetuning_results" / "download_bestcheckpoin.zip"
ADAPTER_DIR  = SCRIPT_DIR / "finetuning_results" / "lora_adapter"
MERGED_DIR   = SCRIPT_DIR / "finetuning_results" / "merged_model"

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Extract LoRA adapter from zip
    print("=" * 60)
    print("Step 1 — Extracting LoRA adapter")
    print("=" * 60)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(ADAPTER_DIR)
    extracted = [p.name for p in ADAPTER_DIR.iterdir()]
    print(f"  Files: {extracted}")
    print(f"  -> Extracted to: {ADAPTER_DIR}\n")

    # 2. Load base model (fp16, CPU — no GPU required for merging)
    print("=" * 60)
    print("Step 2 — Loading base model (fp16 on CPU)")
    print("  First run downloads ~6 GB from HuggingFace.")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"  -> Base model loaded: {BASE_MODEL}\n")

    # 3. Attach and merge LoRA adapter
    print("=" * 60)
    print("Step 3 — Merging LoRA adapter into base model")
    print("=" * 60)
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    print("  -> Adapter loaded, merging weights ...")
    model = model.merge_and_unload()
    print("  -> Merge complete\n")

    # 4. Save merged model in safetensors format
    print("=" * 60)
    print(f"Step 4 — Saving merged model to: {MERGED_DIR}")
    print("=" * 60)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    saved = [p.name for p in MERGED_DIR.iterdir()]
    print(f"  Saved files: {saved}")
    print("\nDone!")
    print("Next step: convert to GGUF with llama.cpp")
    print(f"  -> model dir: {MERGED_DIR.resolve()}")


if __name__ == "__main__":
    main()
