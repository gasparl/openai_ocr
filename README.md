# opanai\_ocr.py – PDF → Clean Text with OpenAI

A small async pipeline that:

1. extracts text (or OCR-renders scanned pages)
2. chops it into model-friendly chunks
3. asks an OpenAI chat model to drop headers/footers and normalise spelling
4. re-assembles and checkpoints the result.

Good for long, mixed PDFs where some pages are born-digital and others are images.

---

## Quick features

* **Header/footer stripping** via prompt engineering – page numbers and running heads are removed.
* **Large-file aware** – splits to ≈ 800 tokens and streams requests concurrently, respecting TPM limits.
* **Checkpoint/resume** – writes partial output every *N* chunks so you can restart safely.
* **Model selectable** – works with `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
* **Automatic retries** with exponential back-off on 429/network errors.

---

## Requirements

```bash
pip install openai pdfplumber tiktoken nltk pytesseract
```

One-time:

```python
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage

```bash
# simplest – reads input.pdf, writes ocr_extract_<model>_<time>.txt
python opanai_ocr.py

# custom paths / model
python opanai_ocr.py \
  --input thesis.pdf \
  --output thesis_clean.txt \
  --model gpt-4o-2024-05-13
```

Environment variable `OPENAI_API_KEY` (or `~/.config/openai/config.json`) must be set.

Important tunables are listed at the top of the script:

| Name            | Default             | Purpose                       |
| --------------- | ------------------- | ----------------------------- |
| `DEFAULT_MODEL` | `gpt-4o-2024-05-13` | Chat model                    |
| `CHUNK_TOKENS`  | `800`               | Prompt + content per call     |
| `CONCURRENCY`   | `2`                 | Parallel calls                |
| `WRITE_EVERY`   | `5`                 | Checkpoint frequency (chunks) |

---

## Output

Plain-text `.txt`.

---
