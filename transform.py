import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bertLayer import BertLayer
from encoder import Encoder
from decoder import Decoder
from log_manager import logger
from sklearn import preprocessing
from scipy import sparse
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

def processing_LDA(tokens, vocab_length, lda_model):
    word_dist_for_topic = []
    for tok in tokens:
        net_tokens = [0] * vocab_length
        for k in tok[1:-2]:
            net_tokens[k] = 1
        topic_probability_scores = lda_model.transform(sparse.csr_matrix(net_tokens))
        topic_indice = list(topic_probability_scores[0]).index(max(topic_probability_scores[0]))
        word_dist_for_topic_raw = lda_model.components_[topic_indice] * topic_probability_scores[0][topic_indice]
        word_dist_for_topic.append(preprocessing.normalize([word_dist_for_topic_raw]))
    return word_dist_for_topic


class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, bert, rate=0.1):
        super(TransformerModel, self).__init__()

        self.encoder = BertLayer(bert)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, inp_input_ids,inp_token_type_ids,inp_attention_mask, tar_input_ids, enc_padding_mask, look_ahead_mask, dec_padding_mask, target_vocab_size, lda_model , alpha=0.1):
        inp = dict()
        inp["input_ids"] = inp_input_ids
        inp["token_type_ids"] = inp_token_type_ids
        inp["attention_mask"] = inp_attention_mask
        tar = dict()
        tar["input_ids"] = tar_input_ids

        enc_output = self.encoder(inp)
        dec_output, attention_weights = self.decoder(tar_input_ids, enc_output, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)


        word_dist_for_topic = processing_LDA(inp_input_ids,vocab_length=target_vocab_size,lda_model=lda_model)
        word_dist_for_topic = torch.Tensor(np.array(word_dist_for_topic)).cuda()
        
        # full_topic = []
        # a_topic = []
        # a = []

        # for e in inp.detach().numpy():
        #     a = np.zeros(18)
        #     for el in e:
        #         a = a + self.word2Topic[int(el)]
        #     a_topic.append(a/300)
        # full_topic.append(a_topic)
       
        # topic_arg1 = torch.matmul(torch.tile(word_dist_for_topic.unsqueeze(1), (1, final_output.shape[1], 1)), 
        #                           torch.reshape(full_topic, (final_output.shape[0], 18, 1)).float())
        # topic_arg2 = torch.tile(torch.reshape(topic_arg1, (topic_arg1.shape[0], 1 , topic_arg1.shape[1])), 
        #                         [1, final_output.shape[1], 1])
        
        # topic_arg3 =  topic_arg2 / torch.max(topic_arg2)

        sum_all = final_output * (1-alpha) + alpha*word_dist_for_topic
        return sum_all, enc_output, attention_weights