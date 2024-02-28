import joblib
import re
import pandas as pd
import torch
from config import *
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, inputs,targets):
        super().__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, index):
        inp_input_ids = self.inputs['input_ids'][index]
        inp_token_type_ids = self.inputs['token_type_ids'][index]
        inp_attention_mask = self.inputs['attention_mask'][index]
        tar_input_ids = self.targets['input_ids'][index]
        tar_token_type_ids = self.targets['token_type_ids'][index]
        tar_attention_mask = self.targets['attention_mask'][index]

        return {
            'inp_input_ids' : inp_input_ids,
            'inp_token_type_ids' : inp_token_type_ids,
            'inp_attention_mask' : inp_attention_mask,
            'tar_input_ids' : tar_input_ids,
            'tar_token_type_ids' : tar_token_type_ids,
            'tar_attention_mask' : tar_attention_mask,
        }

def read_data(tokenizer, filetrain='train_pharagraph_full.jl', filetarget='target_strings_full.jl'):
    train_pharagraph = joblib.load(filetrain)
    target_strings = joblib.load(filetarget)

    targets = []
    for ele in target_strings:
        ele = re.sub(r'[\W_]', ' ', ele)
        ele = re.sub(r'[^\w\s]', '', ele)
        ele = re.sub(r'\t\n', '', ele)
        targets.append(ele)

    training_input = []
    for ele in train_pharagraph:
        ele = re.sub(r'[\W_]', ' ', ele)
        ele = re.sub(r'[^\w\s]', '', ele)
        ele = re.sub(r'\t\n', '', ele)
        training_input.append(ele)

    document = pd.Series(training_input)
    summary = pd.Series(targets)

    doc_list = []
    sum_list = []
    for index, doc in enumerate(document):
        if len(doc) < 2000:
            doc_list.append(doc)
            sum_list.append(summary[index])

    # document = pd.Series(doc_list)
    # summary = pd.Series(sum_list)

    # summary = summary.apply(lambda x: tokenizer.special_tokens_map['cls_token'] + x + tokenizer.special_tokens_map['sep_token'])
    # document = document.apply(lambda x: tokenizer.special_tokens_map['cls_token'] + x + tokenizer.special_tokens_map['sep_token'])

    # document_bert = [tokenizer.encode(d, add_special_tokens=True) for d in document]
    # summary_bert = [tokenizer.encode(d, add_special_tokens=True) for d in summary]

    # inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(d) for d in document_bert], batch_first=True, padding_value=tokenizer.pad_token_id)
    # targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(d) for d in summary_bert], batch_first=True, padding_value=tokenizer.pad_token_id)
    inputs = tokenizer(doc_list, return_tensors="pt", padding=True, max_length = encoder_maxlen, truncation = True)
    targets = tokenizer(sum_list, return_tensors="pt", padding=True, max_length = decoder_maxlen, truncation = True)

    dataset = MyDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader, doc_list, sum_list