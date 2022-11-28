#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import pandas as pd
from transformers import pipeline
import io
from io import StringIO
import torch
import numpy as np
import sagemaker

print(f"Cuda available:{torch.cuda.is_available()}")

s3 = boto3.resource('s3')
bucket_name = 'mlops-team-8'
bucket=s3.Bucket(bucket_name)
directory = 'maildir-stg/'
results_directory = "maildir-results"

model_sent = "philschmid/distilbert-base-multilingual-cased-sentiment-2"
model_sum = "sshleifer/distilbart-xsum-12-3"
model_sent_path = f"{model_sent}"
model_sum_path = f"{model_sum}"

nlp_sent = pipeline("sentiment-analysis", model = model_sent_path, tokenizer = model_sent_path, device = 0)
nlp_sum = pipeline("summarization", model = model_sum_path, tokenizer = model_sum_path, device = 0 )


def infer_sentiment(text: str):

        sentiment = nlp_sent(text, top_k = None, max_length = 512, truncation=True)
        
        out_values = [None, None, None]
        for value in sentiment:
            if value["label"] == 'positive':
                out_values[0] = value["score"]
            if value["label"] == 'neutral':
                out_values[1] = value["score"]
            if value["label"] == 'negative':
                out_values[2] = value["score"]
                
        
        out = [out_values[0], out_values[1], out_values[2]]
        return out
    
def infer_summary(text: str):

        n = len(text.split(" "))
        if n > 62*2:
            max_len = n/2
            if max_len > 1024:
                max_len = 512
            summary =  nlp_sum(text, max_length = max_len, truncation=True)[0]['summary_text']
        elif n > 20:
            summary = nlp_sum(text, max_length = round(n/1.5), truncation=True)[0]['summary_text']
        else:
            summary = text
            
        return summary

def do_infer(row):
    text = row["body"]
    sentiments = infer_sentiment(text)
    summary = infer_summary(text)
    row["positive"] = sentiments[0]
    row["neutral"] = sentiments[1]
    row["negative"] = sentiments[2]
    row["summary"] = summary
    return row


all_files = []
for object_summary in bucket.objects.filter(Prefix=directory):
    all_files.append(object_summary.key)
    
all_files = all_files[1:] 
    
    
s3 = boto3.client('s3')


def get_sentiments(sentiment):
    out_values = [None, None, None]
    for value in sentiment:
        if value["label"] == 'positive':
            out_values[0] = value["score"]
        if value["label"] == 'neutral':
            out_values[1] = value["score"]
        if value["label"] == 'negative':
            out_values[2] = value["score"]


    out = [out_values[0], out_values[1], out_values[2]]
    return out

def extract_summary(summ):
    if isinstance(summ, str):
        return summ
    if isinstance(summ, dict):
        return summ['summary_text']
    return summ


n = len(all_files)
print("starting")
for i in range(n):
    email_file = all_files[i]
    obj = s3.get_object(Bucket=bucket.name, Key=email_file)
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    name = df["username"].unique()[0]
    out_path = f'results/emails_{name}_inference.json'
    if os.path.isfile(out_path) == False:
        
        body_nans = df["body"].isna()
        df.loc[body_nans, "body"] = ""
        
        sentiment = nlp_sent(df["body"].tolist(), top_k = None, max_length = 512, truncation=True)
        df["sentiment"] = sentiment
        df["sentiment_res"] = df.sentiment.apply(get_sentiments)
        df[['positive','neutral', 'negative']] = pd.DataFrame(df.sentiment_res.tolist(), index= df.index)
        df = df.drop(["sentiment", "sentiment_res"], axis = 1)

        df["summary"] = df["body"]
        df['len'] = df["body"].str.split().apply(len)
        long_cond = df.len > 512
        medium_cond = (df.len < 512) & (df.len > 104)
        short_cond = (df.len <= 104) & (df.len >= 62)
        longs = df[long_cond]
        mediums = df[medium_cond]
        shorts = df[short_cond]

        longs_sums = nlp_sum(longs["body"].tolist(), max_length = 512, truncation=True)
        mediums_sums = nlp_sum(mediums["body"].tolist(), max_length = 104, truncation=True)
        shorts_sums = nlp_sum(shorts["body"].tolist(), max_length = 32, truncation=True)

        df.loc[long_cond, "summary"] = longs_sums
        df.loc[medium_cond, "summary"] = mediums_sums
        df.loc[short_cond, "summary"] = shorts_sums
        df["summary"] = df.summary.apply(extract_summary)
        
        df.loc[body_nans, ["body", "body_cleansed", "positive", "neutral", "negative", "summary"]] = np.nan

        #df.to_json(f's3://{bucket.name}/{results_directory}emails_{name}_inference.json')
        df.to_json(out_path)
    perc = (i+1)/n*100
    print(f"Progress: {perc:.2f}%")


sess = sagemaker.Session()
s3_path_to_data = sess.upload_data(bucket=bucket_name, 
                                                  path=results_directory, 
                                                  key_prefix=results_directory)    
    
notebook = "mlops-team-8"
sm = boto3.client('sagemaker')
sm.stop_notebook_instance(NotebookInstanceName=notebook)





