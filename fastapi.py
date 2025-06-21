from fastapi import FastAPI
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pydantic import BaseModel

app = FastAPI()

model_path = "/content/drive/MyDrive/Mental_health_AI/data/reddit_mental_health/mental_health_bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class TextInput(BaseModel):
  text: str

#Inference route
@app.post("/predict")
def predict(input: TextInput):
  inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
  inputs = {k: v.to(device) for k, v in input.items()}

  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

  return {"label": int(predicted_class), "meaning": "depressed" if predicted_class == 1 else "not depressed"}