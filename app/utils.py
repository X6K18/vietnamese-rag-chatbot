import torch
import re
from llm import generate_answer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_with_rag_llm(text, tokenizer, model, label_encoder, device, rag, history=None):
    processed = preprocess_text(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    label = label_encoder.inverse_transform([pred])[0]
    docs = rag.search(text, category=label, k=2)
    answer = generate_answer(text, docs, history=history)
    return label, answer, docs