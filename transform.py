import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bertLayer import BertLayer
from encoder import Encoder
from decoder import Decoder
from log_manager import logger

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output, attention_weights


class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, word2Topic, list_topic_count, bert, rate=0.1):
        super(TransformerModel, self).__init__()

        self.encoder = BertLayer(bert)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = nn.Linear(d_model, target_vocab_size)

        self.word2Topic = word2Topic
        self.list_topic_count = list_topic_count
    
    def forward(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, alpha=0):
        enc_output = self.encoder(inp)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        full_topic = []
        a_topic = []
        a = []
        for e in inp.detach().numpy():
            a = np.zeros(18)
            for el in e:
                a = a + self.word2Topic[int(el)]
            a_topic.append(a/300)
        full_topic.append(a_topic)
       
        topic_arg1 = torch.matmul(torch.tile(self.word2Topic.unsqueeze(0), (final_output.shape[0], 1, 1)), 
                                  torch.reshape(full_topic, (final_output.shape[0], 18, 1)).float())
        topic_arg2 = torch.tile(torch.reshape(topic_arg1, (topic_arg1.shape[0], 1 , topic_arg1.shape[1])), 
                                [1, final_output.shape[1], 1])
        
        topic_arg3 =  topic_arg2 / torch.max(topic_arg2)
        sum_all = final_output * (1-alpha) + alpha*topic_arg3
        return sum_all, enc_output, attention_weights