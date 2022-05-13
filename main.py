from fastapi import FastAPI

app = FastAPI()

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#model = pipeline("sentiment-analysis")

model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
#tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
#model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")


#sentiment_query_sentence = get_random_comment(top_comments)
#sentiment = sentiment_model(sentiment_query_sentence)
#f"The following sentence has be classified with this sentiment: {sentiment_query_sentence} == {sentiment}"

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    query_string: str
  

@app.get("/health")
def health():
    return "Service is online."


@app.post("/prediction")
def my_endpoint(request: PredictionRequest):
    return model(request.query_string)

