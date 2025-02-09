from src.components.model_prediction import Prediction
Pred=Prediction()

def Prediction_pipeline(movie_review,tokenizer,bert_model,device)->str:


    sentiment=Pred.predict_sentiment(text=movie_review,tokenizer=tokenizer,model=bert_model,device=device)

    return sentiment