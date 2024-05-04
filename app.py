import os
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from TextSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class SummaryResponse(BaseModel):
    summary: str

@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects users to the documentation page.
    """
    return RedirectResponse(url="/docs")

@app.post("/train")
async def train_model():
    """
    Endpoint to train the text summarization model.
    """
    try:
        os.system("python main.py")
        return {"message": "Training successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during training: {str(e)}")

@app.post("/predict", response_model=SummaryResponse)
async def predict_text(text: str):
    """
    Endpoint to predict text summarization.
    """
    try:
        prediction_pipeline = PredictionPipeline()
        summary = prediction_pipeline.predict(text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)