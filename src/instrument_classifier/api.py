from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
from contextlib import asynccontextmanager
from instrument_classifier.inference import load_model, process_audio_file
from instrument_classifier.data import InstrumentDataset
import traceback
import uvicorn
from loguru import logger

# Configure loguru to write logs to a file
logger.remove()  # Remove the default logger
logger.add("logging/api.log", rotation="100 MB")

logger.info("API logger initialized")

# Get dataset instance for class mapping
dataset = InstrumentDataset(Path("data/processed/train"), Path("data/processed/metadata_train.csv"))

# Load model at startup
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        logger.info("Loading model...")
        model = load_model()
        logger.success("Model loaded successfully")
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        raise
    yield
    # Shutdown
    logger.info("Shutting down API...")
    model = None


app = FastAPI(title="Instrument Classifier API", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint to predict the instrument class from an audio file.
    Accepts audio files and returns the predicted instrument.
    """
    logger.info(f"Received prediction request for file: {file.filename}")

    if not file.filename.lower().endswith((".wav", ".mp3")):
        error_msg = "Only .wav and .mp3 files are supported"
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            logger.debug(f"Saved temporary file to: {temp_path}")

        # Process the audio file
        logger.info("Processing audio file...")
        prediction_idx = process_audio_file(temp_path, model)
        instrument = dataset.idx_to_class.get(prediction_idx, "unknown")

        # Log prediction
        logger.info(f"File '{file.filename}' predicted as: {instrument}")

        # Clean up
        os.unlink(temp_path)
        logger.debug("Temporary file cleaned up")

        return JSONResponse(content={"predicted_instrument": instrument})

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug("Cleaned up temporary file after error")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {"status": "healthy", "model_loaded": model is not None}
    logger.debug(f"Health check: {status}")
    return status


if __name__ == "__main__":
    logger.info("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
