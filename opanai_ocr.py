#!/usr/bin/env python
# opanai_ocr.py  – fixed & renamed easy_ocr()

from __future__ import annotations
import asyncio, base64, json, logging, os, re, sys, time, datetime
from pathlib import Path
from typing import List, Sequence

import fitz                          # PyMuPDF
from openai import AsyncOpenAI
from openai import BadRequestError, OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
import io
from PIL import Image
import random


# ─── Model catalogue & automatic token-bucket limit ────────────────────────────
TPM_LIMITS: dict[str, int] = {
    "gpt-4.1":     30_000,
    "gpt-4o":      30_000,
    "gpt-4o-mini": 200_000,
    "o1":          30_000,
    "o1-mini":     200_000,
    "o1-pro":      30_000,
    "o3-mini":     200_000,
}

DEFAULT_MODEL = "gpt-4o-mini"               # global default
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
TPM_LIMIT  = TPM_LIMITS[MODEL_NAME]


# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────
CONTEXT_WINDOW      = 128_000
IMAGE_TOKEN_FALLOUT = 110           # pessimistic prompt overhead per image
CONCURRENCY_LIMIT   = 5             # reserved for future parallelism
SAVE_EVERY          = 5             # pages before flushing to disk
TAIL_SENTENCES      = 3             # context sentences passed to next page
MAX_RETRIES         = 3             # extra attempts per page

SYSTEM_PROMPT = (
    "The attached image is a section from a book or a long document. "
    "Process the entire section so that all real content is preserved - including clear chapter titles and any relevant body text. "
    "Do NOT include page numbers, running heads, footers, or any elements that are not genuine content. "
    "If the section is empty or contains no meaningful text, return an entirely empty string only - do not write any explanations, meta-comments, or placeholder text. "
    "In addition, slightly modernize the text: update any outdated spelling or archaic words to standard modern English where appropriate, ensuring readability while preserving the original meaning and tone. "
    "Return plain text only."
)


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────────────
# API client
# ──────────────────────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = json.loads(Path("config.json").read_text())["OPENAI_API_KEY"]
    except Exception:
        log.critical("No OPENAI_API_KEY found in environment or config.json")
        raise SystemExit(1)

client = AsyncOpenAI(api_key=api_key)

# ──────────────────────────────────────────────────────────────────────────────
# Token-per-minute limiter
# ──────────────────────────────────────────────────────────────────────────────
class TokenLimiter:
    """Sliding-window token bucket for TPM limits."""
    def __init__(self, tpm_limit: int) -> None:
        self._tpm   = tpm_limit
        self._lock  = asyncio.Lock()
        self._tokens: List[float] = []               # 1 timestamp per committed token

    async def reserve(self, est_tokens: int) -> None:
        """Block until capacity for *est_tokens* exists inside the rolling 60-s window."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = [t for t in self._tokens if now - t < 60]
                if len(self._tokens) + est_tokens <= self._tpm:
                    self._tokens.extend([now] * est_tokens)
                    return
            await asyncio.sleep(0.25)

    async def commit(self, real_tokens: int, reserved: int) -> None:
        """Reconcile the bucket after usage is known."""
        async with self._lock:
            diff = real_tokens - reserved           # ← fixed (was IMAGE_TOKEN_FALLOUT) :contentReference[oaicite:8]{index=8}
            if diff < 0:                            # over-reserved
                for _ in range(min(-diff, len(self._tokens))):
                    self._tokens.pop()
            elif diff > 0:                          # under-reserved
                self._tokens.extend([time.monotonic()] * diff)

tpm_limiter = TokenLimiter(TPM_LIMIT)
sem_api     = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def iter_page_images_b64(pdf_path: Path, dpi: int = 200):
    """Yield each page rendered to PNG then Base64-encoded."""
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72
        mat  = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            yield base64.b64encode(png_bytes).decode()

# Sentence ends = period, exclamation mark, or question mark
_SENT_DELIM_RE = re.compile(r"(?<=[.!?])\s+")

def last_sentences(text: str, n: int = 3, *, fallback_words: int = 40) -> str:
    """
    Return the last *n* sentences from *text*.

    • If the block has fewer than *n* sentences, the whole block is returned.  
    • If **no** sentence-terminating punctuation appears, we return the last
      *fallback_words* words instead, so the caller still gets usable context.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Split on sentence boundaries (handles spaces and newlines after punctuation)
    sentences = [s for s in re.split(_SENT_DELIM_RE, text) if s]

    if sentences:
        return " ".join(sentences[-n:])
    else:
        # Fallback when no “.” “!” or “?” exists in the block
        return " ".join(text.split()[-fallback_words:])

# ──────────────────────────────────────────────────────────────────────────────
# Core worker
# ──────────────────────────────────────────────────────────────────────────────
async def ocr_page(
    image_b64: str,
    page_num : int,
    prev_tail: str | None = None,
) -> str:
    """Send *one* page (plus context) to GPT-4o Vision and return its text."""
    # Estimate tokens per OpenAI Vision formula: 85 + 170 * tiles
    img_bytes = base64.b64decode(image_b64)
    w, h = Image.open(io.BytesIO(img_bytes)).size
    tiles = -(-w // 512) * -(-h // 512)       # ceiling division
    reserved = int(1.05 * (85 + 170 * tiles))
    
    
    await tpm_limiter.reserve(reserved)

    async with sem_api:
        for attempt in range(MAX_RETRIES + 1):      # bounded retry loop
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if prev_tail:
                    messages.append({"role": "assistant", "content": prev_tail})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            }
                        ],
                    }
                )

                resp = await client.chat.completions.create(
                    model      = MODEL_NAME,
                    # let the model decide the best completion length
                    temperature=0.0,
                    messages   = messages,
                )


                usage = resp.usage
                await tpm_limiter.commit(
                    usage.prompt_tokens + usage.completion_tokens,
                    reserved,
                )
                return resp.choices[0].message.content or ""

            except (BadRequestError, RateLimitError, APIConnectionError,
                    APITimeoutError, OpenAIError) as exc:
                # Roll back the *reserved* tokens – the request never succeeded
                await tpm_limiter.commit(0, reserved)
            
                if attempt >= MAX_RETRIES:
                    raise
                backoff = 2 ** attempt + random.random()
                log.warning("Page %d failed (%s) – retrying in %.1f s (%d/%d)",
                            page_num, exc, backoff, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(backoff)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    pdf_path : Path,
    out_path : Path | None = None,
    *,
    dpi: int = 200,
) -> str:
    """OCR a PDF *sequentially*, saving progress every `SAVE_EVERY` pages."""
    transcript, buffer_pages = [], []
    prev_tail : str | None   = None
    total_pages              = fitz.open(pdf_path).page_count   # cheap metadata call

    for pageno, image_b64 in enumerate(iter_page_images_b64(pdf_path, dpi=dpi), 1):
        page_text = await ocr_page(image_b64, pageno, prev_tail)
        transcript.append(page_text)
        buffer_pages.append(page_text)
        prev_tail = last_sentences(page_text)

        log.info("Processed page %d / %d", pageno, total_pages)
            
        if out_path and pageno % SAVE_EVERY == 0:
            # append last 5 pages to the *main* file
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(" ".join(buffer_pages))
            buffer_pages.clear()
        
            # also write full checkpoint with everything so far
            checkpoint = out_path.with_name(f"{out_path.stem}_sofar_page{pageno}.txt")
            with open(checkpoint, "w", encoding="utf-8") as cp:
                cp.write(" ".join(transcript) + "\n")
        
            log.info("Progress saved through page %d → %s  (checkpoint → %s)",
                     pageno, out_path, checkpoint)


    if out_path and buffer_pages:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(" ".join(buffer_pages))
        log.info("Final pages written → %s", out_path)

    return " ".join(transcript)


# ──────────────────────────────────────────────────────────────────────────────
# Public convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────
import threading
from pathlib import Path

async def easy_ocr_async(
    pdf_path: str | Path,
    out_path: str | Path | None = None,
    *,
    dpi: int = 200,
) -> str:
    _pdf = Path(pdf_path)
    if out_path is None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        _out = _pdf.with_name(f"{_pdf.stem}_{ts}.txt")
    else:
        _out = Path(out_path)
    return await run_pipeline(_pdf, _out, dpi=dpi)



def easy_ocr(
    pdf_path: Path | str = "input.pdf",
    out_path: Path | str | None = None,
    *,
    dpi: int = 200,
) -> str:
    """
    Synchronous helper.

    • In a regular script (no loop running) it simply blocks with `asyncio.run()`.
    • In an interactive session *with* a running loop it transparently spins up
      a worker thread, runs the coroutine there, waits, and finally returns the
      text – so **one plain call does the work everywhere**.
    """
    try:
        asyncio.get_running_loop()             # Is a loop already running?
    except RuntimeError:
        # No → classic blocking path
        return asyncio.run(easy_ocr_async(pdf_path, out_path, dpi=dpi))

    # Yes → run in a background thread instead of nesting loops
    result_holder: list[str] = []
    error_holder: list[BaseException] = []

    def _worker() -> None:
        try:
            result_holder.append(
                asyncio.run(easy_ocr_async(pdf_path, out_path, dpi=dpi))
            )
        except BaseException as e:
            error_holder.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if error_holder:                            # re-raise inside caller thread
        raise error_holder[0]
    return result_holder[0] if result_holder else ""



# ──────────────────────────────────────────────────────────────────────────────
# Simple CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Sequence[str]):
    import argparse
    p = argparse.ArgumentParser(description="OCR a PDF with GPT-4o Vision.")
    p.add_argument("--input", "-i", default="input.pdf", help="Input PDF file")
    p.add_argument("--output", "-o", help="Output text file")
    p.add_argument("--dpi", type=int, default=200, help="Render DPI (default 200)")
    p.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        choices=list(TPM_LIMITS.keys()),
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )    
    return p.parse_args(argv)

def _cli_entry(argv: Sequence[str]) -> None:
    args = _parse_args(argv)
    # Bind the chosen model & TPM limit to the globals other code expects+    
    global MODEL_NAME, TPM_LIMIT, tpm_limiter, sem_api
    MODEL_NAME = args.model
    TPM_LIMIT  = TPM_LIMITS[MODEL_NAME]
    tpm_limiter = TokenLimiter(TPM_LIMIT)
    sem_api     = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    pdf_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        out_path = pdf_path.with_name(f"{pdf_path.stem}_{ts}.txt")

    asyncio.run(run_pipeline(pdf_path, out_path, dpi=args.dpi))


# ──────────────────────────────────────────────────────────────────────────────
# __main__
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli_entry(sys.argv[1:])
    else:
        log.info(
            "opanai_ocr module imported (no CLI args) – ready for easy_ocr()."
        )
