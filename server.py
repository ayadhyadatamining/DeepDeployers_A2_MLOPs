from fastapi import FastAPI, HTTPException, Request, Form ,BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import random
import torch
from transformers import BartForSequenceClassification, BartTokenizer
import os

import uvicorn

from src.pipeline.model_prediction_pipeline import Prediction_pipeline
from src.pipeline.model_training_pipeline import Model_train_pipeline , load_best_model_Parameters
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the fine-tuned model in Google Drive
model_path=os.path.join(os.getcwd(),"Final_trained_model","content","fine_tuned_bart")
print(model_path)

# Load the fine-tuned model and tokenizer
model = BartForSequenceClassification.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)
model.to(device)  # Move model to GPU if available
model.eval()  # Set model to evaluation mode




app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PredictionRequest(BaseModel):
    text: str


class Training_request(BaseModel):
    file_path_positive_review: str
    file_path_negative_review: str
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request, "text": None, "sentiment": None})

@app.post("/analyze_sentiment", response_class=JSONResponse)  # Change response class to JSONResponse
async def analyze_sentiment(request: Request, text: str = Form(...)):
    sentiment_result = Prediction_pipeline(movie_review=text,tokenizer=tokenizer,bert_model=model,device=device)
    #sentiment_result = random.choice(sentiments)  # Randomly choose a sentiment for testing
    print(f"Received text: {text}, Sentiment: {sentiment_result}")  # Debugging line
    return JSONResponse(content={"text": "After Analysis We Can Conclude: ", "sentiment": sentiment_result})  # Return JSON response

@app.get("/model_parameters")
async def get_model_parameters():
    model_parameters = load_best_model_Parameters()
    return JSONResponse(content=model_parameters)

@app.post("/model_training")
async def trigger_model_training(data_file_path: Training_request):

    BackgroundTasks.add_task(Model_train_pipeline,data_file_path.file_path_positive_review,data_file_path.file_path_negative_review)

    return JSONResponse(content={"message": "Model training triggered and added to Background Tasks"})

if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
