import os
from transform import TransformerModel
from CustomSchedule import CustomSchedule
from config import *
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# from transformers import TFBertModel
# model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
import joblib 

import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

if __name__ == "__main__":
    word2Topic = joblib.load('word2Topic.jl')
    list_topic_count = joblib.load('list_topic_count.jl')
    checkpoint_path = "checkpoints"
    learning_rate = CustomSchedule(d_model)
    encoder_vocab_size = tokenizer.vocab_size
    decoder_vocab_size = tokenizer.vocab_size
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # Fixed typo here
    transformer = TransformerModel(
        num_layers, 
        d_model, 
        num_heads, 
        dff,
        encoder_vocab_size, 
        decoder_vocab_size, 
        pe_input=encoder_vocab_size, 
        pe_target=decoder_vocab_size,
        word2Topic=word2Topic,
        list_topic_count=list_topic_count
        )
    transformer.load_weights(checkpoint_path + "/")