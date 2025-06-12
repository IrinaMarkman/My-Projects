from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_local_label(text: str) -> str:
    prompt = (
        "You are a sentiment analysis assistant. Your task is to classify the sentiment "
        "of the text as either 'positive' or 'negative'. Do not use any other labels.\n\n"
        "Examples:\n"
        "Text: I love this product, it works perfectly!\nSentiment: positive\n\n"
        "Text: This is the worst experience I've ever had.\nSentiment: negative\n\n"
        "Text: I feel good about this.\nSentiment: positive\n\n"
        "Text: The service was terrible and slow.\nSentiment: negative\n\n"
        f"Text: {text}\n"
        "Sentiment:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0,
        do_sample=False,
        num_beams=5,
        early_stopping=True
    )
    
    label = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # На всякий случай — если модель сгенерирует что-то другое
    if label not in {"positive", "negative"}:
        label = "negative"

    return label