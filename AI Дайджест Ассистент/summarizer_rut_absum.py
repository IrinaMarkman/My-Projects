from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "cointegrated/rut5-base-absum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def get_summary_rut_absum(text: str) -> str:
    prompt = "Резюмируй текст: " + text
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Если модель повторяет инструкцию, убираем её
    if summary.lower().startswith("резюмируй текст:"):
        summary = summary[len("резюмируй текст:"):].strip()
    
    # Если по-прежнему некорректно, можно вернуть более короткий или заглушку
    if len(summary) < 5:
        summary = "Краткое содержание не может быть сгенерировано."
    
    return summary