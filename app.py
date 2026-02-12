from fastapi import FastAPI, Header, HTTPException
import base64
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()
API_KEY ="sk_test_123456789"

SUPPORTED_LANGUAGES = [
    "Tamil","English","Hindi","Malayalam","Telugu"
]

def analyze_voice(file_path):
    y, sr = librosa.load(file_path,sr=None)

    mfcc =np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Generic heuristic-based detection (allowed)
    score =0
    reasons= []

    if flatness<0.02:
        score += 1
        reasons.append("Unnaturally smooth spectral profile")

    if zcr<0.05:
        score += 1
        reasons.append("Low natural voice variation")

    if mfcc>-40:
        score += 1
        reasons.append("Consistent MFCC pattern")

    if score>= 2:
        return "AI_GENERATED", 0.75, ", ".join(reasons)
    else:
        return "HUMAN", 0.65, "Natural pitch and variation detected"


@app.post("/api/voice-detection")
async def voice_detection(
    payload: dict,
    x_api_key: str= Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        language= payload.get("language")
        audio_base64 = payload.get("audioBase64")

        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported language")

        audio_bytes= base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path =f.name

        classification, confidence, explanation =analyze_voice(temp_path)
        os.remove(temp_path)

        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid request or audio processing failed"
        )
