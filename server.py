#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 11 13:23:10 2020
@author: luc michalski
"""
import argparse
import glob
import os
from os import path
import json
import time
import logging
import random
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import gdown
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request

# Script arguments can include path of the config
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gid', type=str, default="1vhsDOW9wUUO83IQasTPlkxb82yxmMH-V")
arg_parser.add_argument('--host', type=str, default="0.0.0.0")
arg_parser.add_argument('--port', type=str, default="6011")
arg_parser.add_argument('--log', type=str, default="../logs/ovh-qa.log")
args = arg_parser.parse_args()

output = '/root/.cache/t5_que_gen.zip'
output_dir = '/root/.cache/t5_que_gen'
url = 'https://drive.google.com/uc?id='+args.gid

if not path.exists(output):
    print("Downloading the model")
    gdown.download(url, output, quiet=False)

if not path.exists(output_dir) and path.exists(output):
    print("Extracting the model")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        zip_ref.close()

class QueGenerator():
  def __init__(self):
    self.que_model = T5ForConditionalGeneration.from_pretrained(output_dir+'/t5_que_gen_model/t5_base_que_gen/')
    self.ans_model = T5ForConditionalGeneration.from_pretrained(output_dir+'/t5_ans_gen_model/t5_base_ans_gen/')

    self.que_tokenizer = T5Tokenizer.from_pretrained(output_dir+'/t5_que_gen_model/t5_base_tok_que_gen/')
    self.ans_tokenizer = T5Tokenizer.from_pretrained(output_dir+'/t5_ans_gen_model/t5_base_tok_ans_gen/')
    
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    self.que_model = self.que_model.to(self.device)
    self.ans_model = self.ans_model.to(self.device)
  
  def generate(self, text):
    answers = self._get_answers(text)
    questions = self._get_questions(text, answers)
    output = [{'answer': ans, 'question': que} for ans, que in zip(answers, questions)]
    return output
  
  def _get_answers(self, text):
    # split into sentences
    sents = sent_tokenize(text)

    examples = []
    for i in range(len(sents)):
      input_ = ""
      for j, sent in enumerate(sents):
        if i == j:
            sent = "[HL] %s [HL]" % sent
        input_ = "%s %s" % (input_, sent)
        input_ = input_.strip()
      input_ = input_ + " </s>"
      examples.append(input_)
    
    batch = self.ans_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.ans_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                # do_sample=False,
                                # num_beams = 4,
                                )
    dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    answers = [item.split('[SEP]') for item in dec]
    answers = chain(*answers)
    answers = [ans.strip() for ans in answers if ans != ' ']
    return answers
  
  def _get_questions(self, text, answers):
    examples = []
    for ans in answers:
      input_text = "%s [SEP] %s </s>" % (ans, text)
      examples.append(input_text)
    
    batch = self.que_tokenizer.batch_encode_plus(examples, max_length=512, pad_to_max_length=True, return_tensors="pt")
    with torch.no_grad():
      outs = self.que_model.generate(input_ids=batch['input_ids'].to(self.device), 
                                attention_mask=batch['attention_mask'].to(self.device), 
                                max_length=32,
                                num_beams = 4)
    dec = [self.que_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    return dec

app = Flask(__name__)
print("Loading the model")
que_generator = QueGenerator()

@app.route('/query', methods=['POST'])
def query():
    if request.args.get('text'):
        text = request.args.get('text')
    else:
        result = {"status": 400, "msg": "Question cannot be empty"}
        return jsonify(result)

    result = que_generator.generate(text)
    app.logger.info('result: %s', result)
    return jsonify(result)

if __name__ == '__main__':
    handler = RotatingFileHandler(args.log, maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    print("Starting the server")
    app.run(host=args.host, port=args.port)
