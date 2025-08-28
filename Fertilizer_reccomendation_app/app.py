import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# A dictionary to hold our models
ml_models = {}

# Use the lifespan manager to load the model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the machine learning model and label encoder
    print("Loading model and label encoder...")
    ml_models["model"] = joblib.load('fertilizer_artifacts/model.joblib')
    ml_models["label_encoder"] = joblib.load('fertilizer_artifacts/label_encoder.joblib')
    print("Model and label encoder loaded successfully.")
    yield
    # Clean up the ML models and release the resources
    print("Clearing models...")
    ml_models.clear()

# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="Fertilizer Recommendation API",
    description="An API to recommend the best fertilizer based on soil and crop data.",
    version="1.0.0",
    lifespan=lifespan
)

# Define the input data model using Pydantic
class FertilizerData(BaseModel):
    Temperature: int
    Moisture: int
    Rainfall: float
    PH: float
    Nitrogen: int
    Phosphorous: int
    Potassium: int
    Carbon: float
    Soil: str
    Crop: str

# Define the root endpoint to serve the HTML file
@app.get("/", response_class=FileResponse)
def read_root():
    return "index.html"

# Define the prediction endpoint
@app.post("/predict")
def predict_fertilizer(data: FertilizerData):
    # Retrieve the loaded model and encoder from our dictionary
    model = ml_models.get("model")
    label_encoder = ml_models.get("label_encoder")

    if not model or not label_encoder:
        return {"error": "Model not loaded. Cannot make predictions."}

    # Convert the incoming Pydantic data object to a dictionary
    input_data = data.dict()
    
    # Create a pandas DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])
    
    # Make a prediction
    predicted_label = model.predict(input_df)
    
    # Inverse transform the label to get the actual fertilizer name
    predicted_fertilizer = label_encoder.inverse_transform(predicted_label)
    
    # Return the result as JSON
    return {"recommended_fertilizer": predicted_fertilizer[0]}

# This block is for local testing and is ignored by Hugging Face
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)