from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
from inference import load_model, process_audio_file

app = FastAPI(title="Instrument Classifier API")

# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint to predict the instrument class from an audio file.
    Accepts audio files and returns the predicted instrument class.
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
        prediction = process_audio_file(temp_path, model)
        
        # Clean up
        os.unlink(temp_path)
        
        return JSONResponse(content={"predicted_class": int(prediction)})
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}
