import json
from transformers import pipeline


model_sent = "philschmid/distilbert-base-multilingual-cased-sentiment-2"
model_sum = "sshleifer/distilbart-xsum-12-3"
model_sent_path = f"package/{model_sent}"
model_sum_path = f"package/{model_sum}"

nlp_sent = pipeline("sentiment-analysis", model = model_sent_path, tokenizer = model_sent_path)
nlp_sum = pipeline("summarization", model = model_sum_path, tokenizer = model_sum_path)

def lambda_handler(event, context):
    
    text = event['text']

    sentiment = nlp_sent(text, top_k = None)

    n = len(text.split(" "))
    if n > 62*2:
        summary =  nlp_sum(text)[0]['summary_text']
    elif n > 10:
        summary = nlp_sum(text, max_length = round(n/1.5))[0]['summary_text']
    else:
        summary = text

    ans = {"summary" : summary, "sentiment" : sentiment}
    return {
        'statusCode' : 200,
        'body' : ans
    }
