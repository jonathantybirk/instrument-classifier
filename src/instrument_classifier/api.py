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

# Get dataset instance for class mapping
dataset = InstrumentDataset(Path("data/processed/train"), Path("data/raw/metadata_train.csv"))

# Load model at startup
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    yield
    # Shutdown
    model = None

app = FastAPI(title="Instrument Classifier API", lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint to predict the instrument class from an audio file.
    Accepts audio files and returns the predicted instrument.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process the audio file
        prediction_idx = process_audio_file(temp_path, model)
        instrument = dataset.idx_to_class.get(prediction_idx, "unknown")
        
        # Print prediction for debugging
        print(f"File '{file.filename}' predicted as: {instrument}")
        
        # Clean up
        os.unlink(temp_path)
        
        return JSONResponse(content={"predicted_instrument": instrument})
    
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will show in pytest with -s flag
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
