import os
import uuid
import fitz
import asyncio
import edge_tts
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from groq import Groq

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY missing in .env file")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="PDF to Podcast Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Folders
# -------------------------
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
# Generate Podcast Script
# -------------------------
def generate_script(text):
    # Limit input to avoid overflow
    text = text[:12000]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a SOLO expert podcast narrator. Summarize the content into a structured solo speech podcast script. "
                    "Speak entirely in the first-person as ONE single continuous narrator. "
                    "CRITICAL: DO NOT include guests, co-hosts, or an interview format. "
                    "DO NOT use labels like 'Host:', 'Guest:', 'Speaker 1:', or 'Narrator:'. It must be a continuous single-voice monologue."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    script = response.choices[0].message.content

    # Clean script for TTS
    script = script.replace("*", "").replace("#", "")
    return script

# -------------------------
# Generate Audio with Edge TTS
# -------------------------
async def generate_audio(script):
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
        # Save uploaded file
        with open(temp_pdf, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text
        text = extract_text(temp_pdf)

        if not text:
            raise Exception("PDF has no readable text")

        # Generate podcast script
        script = generate_script(text)

        # Generate audio
        audio_filename = await generate_audio(script)

        return JSONResponse({
            "status": True,
            "script": script,
            "audio_file": audio_filename
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": False,
            "message": str(e)
        }, status_code=500)

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

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )
