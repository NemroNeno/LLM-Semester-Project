# qa_from_markdown_openai.py
#
# Prereqs:
# 1) Run qa_extractor.py first to produce sheets_markdown/ folder
# 2) pip install openai>=1.60.0 python-dotenv

import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

MARKDOWN_DIR = "sheets_markdown"   # input folder with per-sheet .md files
OUT_DIR      = "sheets_qa"         # output folder for per-sheet .json files

# -------------------------------------------------------------
# 1) JSON schema
# -------------------------------------------------------------

def qa_schema() -> Dict[str, Any]:
    return {
        "name": "qa_pairs_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "qa_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer":   {"type": "string"},
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["qa_pairs"],
            "additionalProperties": False,
        },
    }

# -------------------------------------------------------------
# 2) Prompts
# -------------------------------------------------------------

def build_system_prompt() -> str:
    return (
        "You are a precise Q/A extraction agent.\n"
        "You receive Markdown content parsed from a single Excel sheet.\n"
        "The sheet contains question and answer pairs.\n\n"
        "Rules:\n"
        "- Extract EVERY question and its corresponding answer.\n"
        "- Copy the text EXACTLY as it appears in the Markdown. "
        "Do NOT paraphrase, summarize, or alter the wording in any way.\n"
        "- If a question spans multiple cells/lines, merge into one string exactly as written.\n"
        "- If an answer spans multiple cells/lines, merge into one string exactly as written.\n"
        "- Do NOT invent questions or answers that are not present.\n"
        "- If no clear answer exists for a question, set answer to empty string \"\".\n"
        "- Exclude table headers, section titles, and metadata.\n\n"
        "Output:\n"
        "- Return a JSON object with key 'qa_pairs'.\n"
        "- Each item: {\"question\": \"...\", \"answer\": \"...\"}.\n"
        "- No extra fields.\n"
    )

def build_user_message(sheet_name: str, markdown: str) -> str:
    return (
        f"Sheet name: {sheet_name}\n\n"
        f"Here is the full Markdown content of this sheet:\n\n"
        f"{markdown}"
    )

# -------------------------------------------------------------
# 3) Call OpenAI for a single markdown (one sheet = one call)
# -------------------------------------------------------------

def call_openai_for_qa(
    sheet_name: str,
    markdown: str,
) -> List[Dict[str, str]]:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": build_user_message(sheet_name, markdown)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": qa_schema(),
        },
        temperature=0.0,
        max_tokens=16_384,   # large enough for all Q/A pairs in one sheet
    )

    finish_reason = response.choices[0].finish_reason
    content = response.choices[0].message.content

    if finish_reason == "length":
        print(f"  [warn] Response truncated for sheet '{sheet_name}'. "
              f"Sheet may be too large for one call.")

    try:
        data = json.loads(content)
        return data.get("qa_pairs", [])
    except json.JSONDecodeError as e:
        print(f"  [error] JSON decode failed for sheet '{sheet_name}': {e}")
        return []

# -------------------------------------------------------------
# 4) Main: iterate all .md files, produce one .json per sheet
# -------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect all markdown files
    md_files = sorted([
        f for f in os.listdir(MARKDOWN_DIR)
        if f.endswith(".md") and f.startswith("sheet_")
    ])

    if not md_files:
        print(f"[error] No sheet_*.md files found in '{MARKDOWN_DIR}'. "
              f"Run qa_extractor.py first.")
        exit(1)

    print(f"[info] Found {len(md_files)} sheet markdown files\n")

    total_pairs = 0

    for md_file in md_files:
        # Derive sheet name from filename: sheet_<SheetName>.md -> <SheetName>
        sheet_name = md_file.removeprefix("sheet_").removesuffix(".md")
        md_path    = os.path.join(MARKDOWN_DIR, md_file)

        print(f"[sheet] '{sheet_name}' ← {md_file}")

        # Read markdown
        with open(md_path, "r", encoding="utf-8") as f:
            markdown = f.read()

        print(f"        {len(markdown)} chars → sending to OpenAI...")

        # Call OpenAI
        qa_pairs = call_openai_for_qa(sheet_name, markdown)
        print(f"        → {len(qa_pairs)} Q/A pairs extracted")

        # Add sheet name to each pair for traceability
        for pair in qa_pairs:
            pair["sheet"] = sheet_name

        # Save JSON for this sheet
        out_json_path = os.path.join(OUT_DIR, f"qa_{sheet_name}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        print(f"        → Saved to {out_json_path}\n")
        total_pairs += len(qa_pairs)

    print(f"[done] Total Q/A pairs across all sheets: {total_pairs}")
    print(f"[done] JSON files saved in: {OUT_DIR}/")
