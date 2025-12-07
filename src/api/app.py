"""
src/api/app.py
---------------
FastAPI service endpoint for the phishing detector pipeline.
Accepts an uploaded .eml file and returns structured extracted features.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


# Logging
from src.utils.logging_utils import configure_logging, get_logger

# Pipeline
from src.inference.pipeline import extract_all


# -------------------------------------------------------------------
# Initialize logging BEFORE creating the FastAPI app
# -------------------------------------------------------------------
configure_logging()
logger = get_logger()


# -------------------------------------------------------------------
# Create FastAPI Application
# -------------------------------------------------------------------
app = FastAPI(
    title="Advanced Spam & Phishing Email Detector",
    description="Multimodal spam and phishing email analyzer (text, URL, and image intelligence).",
    version="1.0.0",
)

app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")
# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.post("/score-email")
async def score_email(file: UploadFile = File(...)):
    """
    Upload a .eml file and return phishing detection results.
    """
    try:
        raw_bytes = await file.read()
        logger.info(
            f"Received file: name={file.filename}, size={len(raw_bytes)} bytes"
        )

        result = extract_all(raw_bytes)

        # Log summary
        hdrs = result.get("headers", {})
        logger.info(
            "Scored email | from={from_} subject={subject!r} label={label} score={score:.3f}",
            from_=hdrs.get("from"),
            subject=hdrs.get("subject"),
            label=result.get("label"),
            score=result.get("phish_score", 0.0),
        )

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        logger.error(f"Error while processing email: {type(e).__name__}: {e}")
        return JSONResponse(
            content={
                "error": f"internal_error:{type(e).__name__}",
                "detail": str(e)
            },
            status_code=500,
        )


@app.get("/")
def home():
    logger.info("Health check called on /")
    return {
        "status": "running",
        "message": "Welcome to the Advanced Spam & Phishing Email Detector API",
        "endpoints": {
            "POST /score-email": "Upload .eml file to analyze email",
        },
    }
