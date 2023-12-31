# -*- coding: utf-8 -*-
"""Memory Retention Task V4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v1x7pKscaG3KGNQPahuKZwUmiFLS7MFy
"""

import pandas as pd
import numpy as np

import torch
import keras
import nltk
import nltk.data
import contractions
import re
import random
import copy
import string
import time, datetime
from tqdm import tqdm
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datasets as ds
import gensim.downloader as api

from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Embedding, LSTM, Dropout, Multiply, Dot, Activation
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.optimizers import Adam, Adagrad, Adadelta

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertModel, BertTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5Model

from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Word2Vec
from gensim.models import KeyedVectors

from nltk import word_tokenize
from nltk.corpus import stopwords, words

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

glove_model = api.load("glove-wiki-gigaword-300")

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import nn
import torch.optim as optim

# git clone https://github.com/facebookresearch/LASER.git

# bash LASER/install_external_tools.sh
# bash LASER/install_models.sh
# bash LASER/nllb/download_models.sh

# mv laser2.pt LASER/nllb/laser2.pt
# mv laser2.spm LASER/nllb/laser2.spm