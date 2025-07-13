# Intelligent PDF OCR & Cleanup with OpenAI

`opanai_ocr.py` is a **one-stop pipeline for turning messy PDFs into clean, modernised plain-text**.
It extracts raw text (or OCRs scanned pages), splits it into token-friendly chunks, sends each chunk to an OpenAI model to strip headers/footers and lightly modernise spelling, and finally stitches everything back together—checkpointing as it goes so you never lose progress.

---

## ✨ Features

* **Smart header/footer removal** – the prompt tells the model to discard page numbers, running heads, etc.
* **Light modernisation** – archaic spelling is updated to contemporary English while preserving meaning.
* **Huge-doc friendly** – splits pages into \~800-token sections and processes them concurrently.
* **Robust rate-limit compliance** – a ticket-based token limiter guarantees you stay within the model’s TPM window, even with many parallel requests.
* **Checkpointing** – writes partial output every *N* sections; if the run aborts you can resume from the last checkpoint.
* **Model-agnostic** – works with `gpt-4o`, `gpt-4-turbo`, or `gpt-3.5-turbo`; you pick.
* **Automatic retries** – backs off and retries on 429s, time-outs, and transient network errors.
* **CLI flags for power users** – choose model, input/output paths, etc.
* **Cross-platform** – tested on macOS, Linux, and Windows 10+.
  (On Windows, the event-loop policy is set automatically for Python ≤3.11.)

---

## 🛠 Requirements

| Package                                          | Purpose                            |
| ------------------------------------------------ | ---------------------------------- |
| `openai`                                         | Chat completion API                |
| `pdfplumber`                                     | Text extraction from PDFs          |
| `tiktoken`                                       | Token counting                     |
| `nltk`                                           | Sentence tokenisation (uses Punkt) |
| **Optional**<br>`pytesseract` + Tesseract engine | Fallback OCR for scanned pages     |

Install everything (except optional OCR) with:

```bash
pip install openai pdfplumber tiktoken nltk
```

### One-time NLTK download

```python
python -c "import nltk; nltk.download('punkt')"
```

---

## ⚙️ Setup

### Add your API key

The script looks for an OpenAI key in either

1. the environment variable `OPENAI_API_KEY`, **or**
2. `~/.config/openai/config.json` (Unix-like) in the form:

   ```json
   { "api_key": "sk-…" }
   ```

---

## 🚀 Usage

Basic:

```bash
python opanai_ocr.py               # uses input.pdf → ocr_extract_<model>_<timestamp>.txt
```

With options:

```bash
python opanai_ocr.py \
    --model gpt-4-turbo-2024-04-09 \
    --input thesis.pdf \
    --output thesis_clean.txt
```

During execution you’ll see log lines such as:

```
12:10:04 | INFO | 42 chunks queued with gpt-4o-2024-05-13
12:10:08 | INFO | Section 3/42 (try 1)
12:10:08 | INFO | TPM cap reached; sleeping 34.2s
12:12:55 | INFO | Checkpoint saved: thesis_clean.p005.txt
```

---

## 📄 Output

* Plain-text `.txt` file.

---

## 🔧 Customisation

| Variable                | Meaning                                               | Default             |
| ----------------------- | ----------------------------------------------------- | ------------------- |
| `DEFAULT_MODEL`         | model used when `--model` not supplied                | `gpt-4o-2024-05-13` |
| `CHUNK_TOKENS`          | soft cap for each section (prompt + content)          | `800`               |
| `MAX_COMPLETION_TOKENS` | reply cap; auto-adjusted downward if context is tight | `256`               |
| `CONCURRENCY`           | simultaneous chunks hitting the API                   | `2`                 |
| `RETRIES`               | attempts per chunk before giving up                   | `5`                 |
| `WRITE_EVERY`           | checkpoint frequency                                  | `5` chunks          |

Feel free to tweak them near the top of the script.

---

## 📝 Example end-to-end

```bash
# Convert a 19th-century scanned book to clean text with modern spelling
python opanai_ocr.py --input 1858_railway_manual.pdf --model gpt-4o-2024-05-13
```

Output:

```
ocr_extract_gpt-4o-2024-05-13_20250713_1412.txt
```

Open it and enjoy a neatly formatted, header-free, easy-to-read version of your original PDF.

---

## 📚 License

MIT – see [LICENSE](LICENSE) for details.

Happy digitising!
