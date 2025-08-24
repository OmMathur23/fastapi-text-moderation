from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ----------------------------
# Pydantic Model
# ----------------------------
class TextInput(BaseModel):
    texts: Union[str, List[str]]

# ----------------------------
# Load Model at Startup
# ----------------------------
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)
# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Text Moderation API")
@app.get("/")
def root():
    return {"message": "Text Moderation API is running 🚀"}
@app.post("/moderate")
def moderate(input: TextInput):
    texts = input.texts if isinstance(input.texts, list) else [input.texts]

    results = classifier(texts) 
    output = []

    for text, result in zip(texts, results):
        categories = [
            {"name": r["label"], "score": float(r["score"])}
            for r in result
        ]
        toxic_score = next((c["score"] for c in categories if c["name"].lower() == "toxic"), 0.0)

        action = "block" if toxic_score > 0.7 else "allow"

        output.append({
            "text": text,
            "action": action,
            "categories": categories,
            "reason": f"Toxicity score {toxic_score:.2f}"
        })

    return {"results": output}
