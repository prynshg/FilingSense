"""Step 6 test: transcribe a slice of the JPM earnings call and index it.

Prereq: download the call audio first, e.g.
    yt-dlp -x --audio-format mp3 -o "data/jpm_call.%(ext)s" "<youtube-url>"
Any JPM earnings-call upload works; we only transcribe the first 10 minutes.
"""
from filingsense.audio.transcribe import chunk_transcript, transcribe
from filingsense.retrieval.hybrid import HybridRetriever

AUDIO = "data/jpm_call.webm"

segments = transcribe(AUDIO, max_minutes=10.0)
print(f"transcribed {len(segments)} segments, covering {segments[-1]['end']:.0f}s")
print("sample:", segments[0]["text"][:150])

chunks = chunk_transcript(segments)
print(f"chunked into {len(chunks)} transcript chunks")

retriever = HybridRetriever()
retriever.upsert(chunks)  # embeds + indexes alongside the 10-K chunks
print("transcript chunks indexed (source=earnings_call)")

print("\n=== What did management say about the economic outlook? ===")
for i, c in enumerate(retriever.search("management economic outlook guidance", top_k=3)):
    md = c["metadata"]
    print(f"[{i+1}] ({md.get('source')}, {md.get('start_sec', '?')}s): "
          f"{c['text'][:200].replace(chr(10), ' ')}...")
