

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
 
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore')


def evaluated(encoder, decoder):
    print("대화를 시작합니다.\n")
 
    for i in range(10):
        print("끝내고 싶으면 '종료'를 입력하세요\n")
        inp = input()
        if inp == "종료":
            break
        output_words, attentions = evaluate(encoder, decoder , inp)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

df = pd.read_csv("ChatBotData.csv")

use_cuda = torch.cuda.is_available()
 
MAX_LENGTH = 20

SOS_token = 0
EOS_token = 1
UNKNOWN_token = 2
class Lang :
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.word2count = {0: "SOS", 1: "EOS", 2:"UNKNOWN"}
        self.n_words = 3 #count SOS and EOS and UNKWON
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalizeString(s):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣 ^☆; ^a-zA-Z.!?]+')
    result = hangul.sub('', s)

    return s

def readText():
    inputs = df['Q']
    outputs = df['A']
 
    inputs = [normalizeString(s) for s in inputs]
    outputs = [normalizeString(s) for s in outputs]   
    inp = Lang('input')
    outp = Lang('output')   
    pair = []
    for i in range(len(inputs)):
        pair.append([inputs[i], outputs[i]])
    return inp, outp, pair


def prepareData():
    input_lang, output_lang, pairs = readText()

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs
 
input_lang, output_lang, pairs = prepareData()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        result = torch.zeros(1,1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2 , self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]) , 1 )))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights
    
    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        

def indexesFromSentence(lang, sentence):
     return [lang.word2index[word] for word in sentence.split(' ')]
 

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)

    result = torch.LongTensor(indexes).view(-1,1)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

teacher_forcing_ratio = 0.5



input_lang

input_lang.index2word[1] = '모르는 단어'
input_lang.word2index['모르는 단어'] = 1

def indexesFromSentence(lang, sentence):
    ls = []
    for word in sentence.split(' '):
        try:
            ls.append(lang.word2index[word])
        except:
            ls.append(lang.word2index['모르는 단어'])
    return ls 

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

checkpoint = torch.load(
    'model.pt', map_location=lambda storage, loc: storage)

encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']

encoder1.load_state_dict(encoder_sd)

attn_decoder1.load_state_dict(decoder_sd)

encoder1.state_dict()

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:

            decoded_words.append(output_lang.index2word[ni.item()])
        
        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        
    return decoded_words, decoder_attentions[:di +1]

evaluated(encoder1, attn_decoder1)

