import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_angles(position, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
    return position * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        torch.arange(position)[:, None],
        torch.arange(d_model)[None, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads.unsqueeze(0)

    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

def beam_search_decoder(predictions, top_k=3):
    output_sequences = [([], 0)]

    for token_probs in predictions:
        new_sequences = []

        for old_seq, old_score in output_sequences:
            for char_index, token_prob in enumerate(token_probs):
                new_seq = old_seq + [char_index]
                new_score = old_score + torch.log(token_prob)
                new_sequences.append((new_seq, new_score))

        output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)
        output_sequences = output_sequences[:top_k]

    return output_sequences

def loss_function(real, pred):
    loss_object = nn.CrossEntropyLoss(ignore_index=0)
    mask = (real != 0)
    loss_ = loss_object(pred.permute(0, 2, 1), real)

    mask = mask.float()
    loss_ *= mask
    final_loss = torch.sum(loss_) / torch.sum(mask)
    return final_loss

def evaluate(input_document, tokenizer, encoder_maxlen, decoder_maxlen, transformer):
    input_document = tokenizer.special_tokens_map['cls_token'] + input_document + tokenizer.special_tokens_map['sep_token']
    input_document = [tokenizer(d, return_tensors='pt')['input_ids'].numpy().tolist()[0] for d in input_document]
    input_document = torch.tensor(input_document, dtype=torch.int32)
    encoder_input = input_document[0].unsqueeze(0)
    decoder_input = tokenizer(tokenizer.special_tokens_map['cls_token'], return_tensors='pt')['input_ids'].numpy().tolist()[0]
    output = torch.tensor(decoder_input, dtype=torch.int32).unsqueeze(0)
    
    get_final_value = torch.zeros(1, 1, tokenizer.vocab_size)
    for _ in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        
        predictions, attention_weights, _ = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )
        predictions = predictions[:, -1:, :]
        get_final_value = torch.cat([get_final_value, predictions])
        predicted_id = torch.argmax(predictions, dim=-1)

        if predicted_id.item() == tokenizer("stop", return_tensors='pt')['input_ids'].numpy().tolist()[0][1]:
            return output.squeeze(0), get_final_value, attention_weights 

        output = torch.cat([output, predicted_id], dim=-1)

    return output.squeeze(0), get_final_value, attention_weights

def create_padding_mask(seq):
    seq = (seq == 0)
    return seq.unsqueeze(1).unsqueeze(1)

def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    print(enc_padding_mask.is_cuda())
    print(dec_padding_mask.is_cuda())
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    print(look_ahead_mask.is_cuda())
    dec_target_padding_mask = create_padding_mask(tar)
    print(dec_target_padding_mask.is_cuda())

    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask