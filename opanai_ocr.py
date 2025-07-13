#!/usr/bin/env python3
"""
PDF-to-Text extractor with OpenAI post-processing.
- Extracts text from a PDF.
- Splits and cleans up chunks.
- Sends chunks to OpenAI for "intelligent OCR" (removes headers, etc.).
- Outputs clean text with slightly modernized language.
"""

import os
import re
import asyncio
import json
import logging
import time
import random
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from collections import deque
from typing import List

import pdfplumber
import tiktoken
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIConnectionError, APITimeoutError

# ---------- Model list & settings ----------
CONTEXT_WINDOWS = {
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-3.5-turbo-0125": 16_384,
}
TPM_LIMITS = {
    "gpt-4o-2024-05-13": 30_000,
    "gpt-4-turbo-2024-04-09": 30_000,
    "gpt-3.5-turbo-0125": 180_000,
}
AVAILABLE_MODELS = list(CONTEXT_WINDOWS.keys())
DEFAULT_MODEL = "gpt-4o-2024-05-13"
WINDOW = 60
CHUNK_TOKENS = 800
MAX_COMPLETION_TOKENS = 256
RETRIES, CONCURRENCY, WRITE_EVERY = 5, 2, 5

SYSTEM_PROMPT = (
    "The following text is a section from a book or a long document. "
    "Process the entire section so that all real content is preserved—including chapter titles, subtitles, and any relevant body text. "
    "Do NOT include page numbers, running heads, footers, or any elements that are not genuine content. "
    "If the section is empty or contains no meaningful text, return an entirely empty string only—do not write any explanations, meta-comments, or placeholder text. "
    "In addition, slightly modernize the text: update any outdated spelling or archaic words to standard modern English where appropriate, ensuring readability while preserving the original meaning and tone. "
    "Your output must contain only the preserved, modernized content, with no added explanations, meta-information, or formatting such as Markdown code fences. Return plain text only."
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ---------- API client ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = json.loads(Path("config.json").read_text())["OPENAI_API_KEY"]
    except Exception:
        log.critical("No OPENAI_API_KEY found in environment or config.json")
        raise SystemExit(1)

client = AsyncOpenAI(api_key=api_key)

# ---------- Token helpers ----------
@lru_cache
def _encoder(model: str = DEFAULT_MODEL):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def token_len(txt: str, model: str = DEFAULT_MODEL) -> int:
    return len(_encoder(model).encode(txt))

def context_window(model: str) -> int:
    return CONTEXT_WINDOWS.get(model, 128_000)

# ---------- PDF Text Extraction ----------
def read_pdf(path: Path) -> List[str]:
    paragraphs = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Split into paragraphs, removing empty/short lines
                for para in re.split(r'\n{2,}', text):
                    clean = para.strip()
                    if len(clean) >= 10:
                        paragraphs.append(clean)
    return paragraphs

def split_paragraphs(paragraphs: List[str], model: str) -> List[str]:
    chunks, current, tokens = [], [], 0
    for p in paragraphs:
        p_tokens = token_len(p, model) + 1
        if p_tokens > CHUNK_TOKENS:
            # Split long paragraph by sentences
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            buf, buf_tokens = [], 0
            for s in sents:
                s_tok = token_len(s, model) + 1
                if buf_tokens + s_tok > CHUNK_TOKENS and buf:
                    chunks.append(" ".join(buf)); buf, buf_tokens = [], 0
                buf.append(s); buf_tokens += s_tok
            if buf: chunks.append(" ".join(buf))
            continue
        if tokens + p_tokens > CHUNK_TOKENS and current:
            chunks.append("\n".join(current)); current, tokens = [], 0
        current.append(p); tokens += p_tokens
    if current: chunks.append("\n".join(current))
    return chunks

def strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text.strip(), flags=re.S)

class TokenLimiter:
    def __init__(self, model: str):
        self.limit = TPM_LIMITS.get(model, 30_000)
        self.window = WINDOW
        self.usage = deque()
        self.lock = asyncio.Lock()

    async def throttle(self, tokens_budget: int):
        async with self.lock:
            now = time.time()
            while self.usage and now - self.usage[0][0] > self.window:
                self.usage.popleft()
            used = sum(t for _, t in self.usage)
            if used + tokens_budget > self.limit:
                sleep = self.window - (now - self.usage[0][0]) + 0.05
                sleep = max(sleep, 0.1)
                log.info(f"TPM cap reached; sleeping {sleep:.1f}s")
                await asyncio.sleep(sleep)
            self.usage.append((time.time(), tokens_budget))

    async def commit(self, real_tokens: int):
        async with self.lock:
            if self.usage:
                self.usage.pop()
            self.usage.append((time.time(), real_tokens))

sem = asyncio.Semaphore(CONCURRENCY)

async def extract(chunk: str, model: str,
                  limiter: TokenLimiter, idx: int, total: int) -> str:
    async with sem:
        prompt_toks = token_len(SYSTEM_PROMPT, model)
        chunk_toks  = token_len(chunk, model)
        avail       = max(0, context_window(model) - prompt_toks - chunk_toks)
        max_comp    = max(32, min(MAX_COMPLETION_TOKENS, avail))

        await limiter.throttle(prompt_toks + chunk_toks + max_comp)
        for attempt in range(RETRIES):
            try:
                log.info(f"Section {idx}/{total} (try {attempt+1})")
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": chunk},
                    ],
                    max_tokens=max_comp,
                    temperature=0.1,
                )
                comp_toks = resp.usage.completion_tokens or max_comp
                await limiter.commit(prompt_toks + chunk_toks + comp_toks)

                text = resp.choices[0].message.content or ""
                return strip_code_fence(text)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == RETRIES - 1:
                    raise
                wait = 2 ** attempt + random.random()
                log.warning(f"{type(e).__name__}: {e}; retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            except OpenAIError:
                raise

async def run_pipeline(input_pdf: Path, output_txt: Path, model: str):
    paras   = read_pdf(input_pdf)
    chunks  = split_paragraphs(paras, model)
    total   = len(chunks)
    log.info(f"{total} chunks queued with {model}")

    out_lines = [(
        f"--- OCR/Extraction Report ---\n"
        f"Model: {model}\n"
        f"Date:  {datetime.now():%Y-%m-%d %H:%M}\n"
        f"Source file: {input_pdf.name}\n"
        f"---\n\n"
    )]

    limiter   = TokenLimiter(model)
    tasks = [extract(ch, model, limiter, i, total)
             for i, ch in enumerate(chunks, 1)]
    for i, result in enumerate(await asyncio.gather(*tasks), 1):
        out_lines.append(f"-- Section {i} --\n{result}\n\n" if result.strip()
                         else f"-- Section {i} --\n\n")

        if i % WRITE_EVERY == 0 or i == total:
            ckpt = output_txt.with_suffix(f".p{i:03d}{output_txt.suffix}")
            ckpt.write_text("".join(out_lines), encoding="utf-8")
            log.info(f"Checkpoint saved: {ckpt.name}")

    output_txt.write_text("".join(out_lines), encoding="utf-8")
    log.info(f"Done → {output_txt}")

def start_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)

def easy_extract(model: str = DEFAULT_MODEL,
                 input_pdf: str = "input.pdf",
                 output_txt: str = None):
    inp   = Path(input_pdf).resolve()
    now   = datetime.now().strftime("%Y%m%d_%H%M")
    out   = Path(output_txt) if output_txt else inp.with_name(
           f"ocr_extract_{model}_{now}.txt")
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"\n→ Extracting {inp.name} with {model}\n")
    start_async(run_pipeline(inp, out, model))
    print(f"\n✓ Output written to {out}\n")

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import argparse
        p = argparse.ArgumentParser(description="Extract PDF text via OpenAI.")
        p.add_argument("--model",  choices=AVAILABLE_MODELS, default=DEFAULT_MODEL)
        p.add_argument("--input",  default="input.pdf", help="PDF file path")
        p.add_argument("--output", help="Output TXT (optional)")
        args = p.parse_args()
        start_async(run_pipeline(Path(args.input),
                             Path(args.output) if args.output else
                             Path(f"ocr_extract_{args.model}_{datetime.now():%Y%m%d_%H%M}.txt"),
                             args.model))
