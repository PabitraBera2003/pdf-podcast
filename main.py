import os
import uuid
import fitz
import asyncio
import time
import re
import edge_tts
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY missing in .env file")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="PDF to Podcast Service (Large Book Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config — tune these if needed
# -------------------------
# Free tier: 6000 TPM. Each call ~800-1200 tokens total.
# 6000 / 1000 = 6 calls/min max → safe = 1 call per 12s. Use 15s.
CALL_DELAY_SECONDS = 3
MAX_CHUNKS_TO_PROCESS = 40   # Cap for large PDFs — 40 is plenty for a podcast
CHUNK_SIZE_WORDS = 1500      # Smaller = fewer input tokens per call
CHUNK_OVERLAP_WORDS = 100

os.makedirs("generated_audio", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# -------------------------
# PDF Text Extraction
# -------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

# -------------------------
# Text Chunking
# -------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

# -------------------------
# Sample chunks evenly across full book
# -------------------------
def sample_chunks(chunks, max_chunks=MAX_CHUNKS_TO_PROCESS):
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    indices = [int(i * step) for i in range(max_chunks)]
    sampled = [chunks[i] for i in indices]
    print(f"  Sampled {max_chunks} chunks evenly from {len(chunks)} total.")
    return sampled

# -------------------------
# Parse retry-after seconds from Groq error message
# -------------------------
def parse_retry_after(err_str):
    match = re.search(r'(\d+)m([\d.]+)s', err_str)
    if match:
        return int(int(match.group(1)) * 60 + float(match.group(2))) + 5
    match = re.search(r'([\d.]+)s', err_str)
    if match:
        return int(float(match.group(1))) + 5
    return None

# -------------------------
# Groq API call with smart retry
# -------------------------
def call_groq(messages, max_tokens=350, temperature=0.3, retries=6):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            err_str = str(e)

            # Request too large — skip this chunk
            if "413" in err_str:
                print("  Chunk too large (413), skipping.")
                return ""

            # Daily limit exhausted — fail fast, no point retrying
            if "tokens per day" in err_str.lower() or "TPD" in err_str:
                raise Exception(
                    "Daily token limit (TPD) reached on your Groq free tier. "
                    "Please wait until tomorrow or upgrade: https://console.groq.com/settings/billing"
                )

            # Per-minute rate limit — wait then retry
            if "429" in err_str or "rate_limit" in err_str.lower():
                retry_after = parse_retry_after(err_str)
                if retry_after:
                    print(f"  Rate limit. Groq says wait {retry_after}s. Waiting...")
                    time.sleep(retry_after)
                else:
                    wait = (2 ** attempt) * 15
                    print(f"  Rate limit. Waiting {wait}s (attempt {attempt+1}/{retries})...")
                    time.sleep(wait)
                continue

            print(f"  Unexpected error: {e}")
            raise e

    raise Exception("Max retries exceeded on Groq API call.")

# -------------------------
# Generate Podcast Script
# -------------------------
def generate_script(text):
    all_chunks = chunk_text(text)
    chunks = sample_chunks(all_chunks, max_chunks=MAX_CHUNKS_TO_PROCESS)

    est_minutes = len(chunks) * CALL_DELAY_SECONDS // 60
    print(f"\nLevel 1: Processing {len(chunks)} chunks (~{est_minutes} min at {CALL_DELAY_SECONDS}s delay)...\n")

    # ── LEVEL 1: Each chunk → bullet points ─────────────────────────────────
    bullet_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        result = call_groq(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract ONLY the 3-5 most important facts, ideas, or events from this text. "
                        "Reply with very short bullet points only. No intro, no filler."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        if result:
            bullet_summaries.append(result)

        if i < len(chunks) - 1:
            time.sleep(CALL_DELAY_SECONDS)

    if not bullet_summaries:
        raise Exception("No bullet points were extracted. Check your Groq API limits.")

    # ── LEVEL 2: All bullets → final podcast script ──────────────────────────
    print(f"\nLevel 2: Generating final podcast script from {len(bullet_summaries)} summaries...")

    all_bullets = "\n\n".join(bullet_summaries)

    # Safety truncation: cap combined bullets at 4000 words
    words = all_bullets.split()
    if len(words) > 4000:
        all_bullets = " ".join(words[:4000])
        print("  Truncated combined bullets to 4000 words.")

    time.sleep(CALL_DELAY_SECONDS)

    script = call_groq(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a solo expert podcast narrator. Convert these book notes into an engaging "
                    "solo podcast script. Speak as ONE continuous narrator in first person. "
                    "Cover the beginning, middle, and end of the book. "
                    "DO NOT use labels like 'Host:', 'Guest:', 'Narrator:'. "
                    "Start directly with the content. Use natural spoken language."
                ),
            },
            {"role": "user", "content": all_bullets},
        ],
        max_tokens=2000,
        temperature=0.7,
    )

    script = script.replace("*", "").replace("#", "").strip()
    print("Script generation complete.")
    return script

# -------------------------
# Generate Audio
# -------------------------
async def generate_audio(script):
    print("Generating audio...")
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = f"generated_audio/{audio_filename}"
    communicate = edge_tts.Communicate(script, "en-US-AriaNeural")
    await communicate.save(audio_path)
    return audio_filename

# -------------------------
# API: Generate Podcast
# -------------------------
@app.post("/generate")
async def generate_podcast(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_pdf = f"temp/{uuid.uuid4()}.pdf"

    try:
        with open(temp_pdf, "wb") as buffer:
            buffer.write(await file.read())

        text = extract_text(temp_pdf)
        if not text:
            raise Exception("PDF has no readable text")

        loop = asyncio.get_event_loop()
        script = await loop.run_in_executor(None, generate_script, text)

        audio_filename = await generate_audio(script)

        return JSONResponse({
            "status": True,
            "script": script,
            "audio_file": audio_filename
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": False, "message": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)

# -------------------------
# API: Stream Audio
# -------------------------
@app.get("/stream/{filename}")
async def stream_audio(filename: str):
    file_path = f"generated_audio/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
