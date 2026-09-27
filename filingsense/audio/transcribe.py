"""Transcribe earnings-call audio with faster-whisper (GPU).

We only transcribe the first `max_minutes` minutes: faster-whisper yields
segments progressively, so breaking out of the loop stops decoding early.
"""
from __future__ import annotations

import json
from pathlib import Path


def _ensure_nvidia_dlls() -> None:
    """Windows: let CTranslate2 find cuBLAS/cuDNN from the pip nvidia wheels.

    Two mechanisms, belt and braces: os.add_dll_directory() plus prepending
    to PATH. Some native loaders ignore the former; plain LoadLibrary always
    searches PATH dynamically at load time.
    """
    import os
    import sys

    if os.name != "nt":
        return
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nv.is_dir():
        return
    dirs = [str(nv / sub) for sub in ("cublas/bin", "cudnn/bin", "cuda_runtime/bin")]
    dirs = [d for d in dirs if Path(d).is_dir()]
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])



def transcribe(
    audio_path: str | Path,
    out_path: str | Path = "data/jpm_call_transcript.json",
    model_size: str = "base",
    max_minutes: float = 10.0,
) -> list[dict]:
    _ensure_nvidia_dlls()  # must run before the engine loads CUDA libs
    from faster_whisper import WhisperModel

    print("transcribing on cuda (float16)", flush=True)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _info = model.transcribe(str(audio_path), beam_size=5)
    out: list[dict] = []
    cutoff = max_minutes * 60
    for s in segments:
        if s.start > cutoff:
            break
        out.append({"start": round(s.start, 1), "end": round(s.end, 1),
                    "text": s.text.strip()})
    out_path = Path(out_path)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def chunk_transcript(
    segments: list[dict],
    *,
    ticker: str = "JPM",
    target_chars: int = 1500,
) -> list[dict]:
    """Group transcript segments into chunks, keeping timestamps in metadata."""
    chunks: list[dict] = []
    buf: list[dict] = []
    buf_chars = 0
    idx = 0

    def flush() -> None:
        nonlocal idx, buf, buf_chars
        if not buf:
            return
        text = " ".join(s["text"] for s in buf)
        chunks.append({
            "id": f"{ticker.lower()}_call_{idx:03d}",
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source": "earnings_call",
                "chunk_id": idx,
                "start_sec": buf[0]["start"],
                "end_sec": buf[-1]["end"],
                "has_tables": False,
            },
        })
        idx += 1
        buf = []
        buf_chars = 0

    for s in segments:
        buf.append(s)
        buf_chars += len(s["text"])
        if buf_chars >= target_chars:
            flush()
    flush()
    return chunks
