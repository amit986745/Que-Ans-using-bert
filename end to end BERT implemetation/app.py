import streamlit as st
import tensorflow as tf 
import requests
import json
import torch
import os
from tqdm import tqdm
import string, re
from transformers import BertForQuestionAnswering, BertTokenizerFast

# @st.cache(allow_output_mutation=True) 


model_path="C:\\Users\\AMIT PAREEK\\Downloads\\Q and A using Bert\\end to end BERT implemetation"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')
model = model.to(device)
def get_prediction(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs[0])  
    answer_end = torch.argmax(outputs[1]) + 1 
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
  
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return round(2 * (prec * rec) / (prec + rec), 2)
  
def question_answer(context, question):
    prediction = get_prediction(context,question)
    # em_score = exact_match(prediction, answer)
    # f1_score = compute_f1(prediction, answer)
    return prediction
    # print(f'Question: {question}')
    # print(f'Prediction: {prediction}')
    # print(f'True Answer: {answer}')
    # print(f'Exact match: {em_score}')
    # print(f'F1 score: {f1_score}\n')
st.title("Ask question based on your Article")
articels=st.text_area("please enter your articel")
quest=st.text_input("Ask question based on your Article")
button=st.button("ANSWERE")

with st.spinner("Finding Answere ->"):
    if button and articels:
        answeres=question_answer(articels, quest)
        st.success(answeres)
        