import torch
from torch.utils.data import DataLoader
from CustomSchedule import CustomSchedule
from loadData import read_data 
from config import *
import joblib
import time
from transform import TransformerModel 
from helper import *
from log_manager import *
from queue import PriorityQueue

from transformers import AutoTokenizer
from transformers import BertModel
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = BertModel.from_pretrained("vinai/phobert-base-v2")

lda_model = joblib.load('../input/summarize-dataset/lda_model.jl')


train_loss = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BeamSearchNode(object):
    def __init__(self, prev_node, token_id, log_prob):
        self.finished = False   # Determine if the hypothesis decoding is finished
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob

        if prev_node is None:
            self.seq_tokens = [token_id]
        else:
            self.seq_tokens = prev_node.seq_tokens + [token_id]

        self.seq_len = len(self.seq_tokens)

        if token_id == tokenizer.eos_token_id:
            self.finished = True

    def eval(self):
        alpha = 1.0
        reward = 0

        # Add here a function for shaping a reward
        return self.log_prob / float(self.seq_len - 1 + 1e-6) + alpha * reward

def evaluate_beam(input_document, n_best, k_beam, transformer):
    input_document = [tokenizer.encode(d, add_special_tokens=True) for d in input_document]
    input_document = torch.nn.utils.rnn.pad_sequence([torch.tensor(d) for d in input_document], batch_first=True, padding_value=tokenizer.pad_token_id)
    input_document = input_document.to(device)
    encoder_input = input_document[0].unsqueeze(0)
    decoder_input = torch.tensor(tokenizer.encode(tokenizer.special_tokens_map['cls_token'], add_special_tokens=True)).unsqueeze(0).to(device)
    output = decoder_input.unsqueeze(0)
    decoded_batch = []
    beam_hypotheses = []
    start_node = BeamSearchNode(prev_node=None, token_id=tokenizer.special_tokens_map['cls_token'], log_prob=0)
    beam_hypotheses.append((-start_node.eval(), start_node))
    end_nodes = []

    for i in range(decoder_maxlen):
        candidates = PriorityQueue()
        for score, node in beam_hypotheses:
            dec_seq = node.seq_tokens
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            predictions, attention_weights, _ = transformer(
                encoder_input, 
                output,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask
            )
            predictions = predictions[: ,-1:, :]
            sorted_probs = torch.sort(predictions[0][0], dim=-1, descending=True)[0]
            sorted_indices = torch.sort(predictions[0][0], dim=-1, descending=True)[1]
            for i in range(n_best):
                decoded_token = sorted_indices[i]
                log_prob = sorted_probs[i]
        
                next_node = BeamSearchNode(prev_node=node, token_id=decoded_token, log_prob=node.log_prob + log_prob)
                
                if decoded_token == tokenizer.eos_token_id:
                    end_nodes.append((-next_node.eval(), next_node))
                else:
                    candidates.put((-next_node.eval(), next_node))
            if len(end_nodes) >= k_beam:
                break
            beam_hypotheses = [candidates.get() for _ in range(k_beam)]
    best_hypotheses = []
    sorted_beam_hypotheses = sorted(beam_hypotheses, key=lambda x: x[0])
    
    if len(end_nodes) < k_beam:
        for i in range(k_beam - len(end_nodes)):
            end_nodes.append(sorted_beam_hypotheses[i])
    sorted_end_nodes = sorted(end_nodes, key=lambda x: x[0])

    for i in range(k_beam):
        score, end_node = sorted_end_nodes[i]
        best_hypotheses.append((score, end_node.seq_tokens))
    decoded_batch.append(best_hypotheses)
    return decoded_batch

def train_step(inp_input_ids, inp_token_type_ids, inp_attention_mask, tar_input_ids, target_vocab_size, transformer, optimizer, scheduler):
    tar_real = tar_input_ids
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_input_ids, tar_input_ids)
    predictions, enc_output, att_weights = transformer(
        inp_input_ids, inp_token_type_ids, inp_attention_mask, tar_input_ids, 
        enc_padding_mask, 
        combined_mask, 
        dec_padding_mask,
        target_vocab_size,
        lda_model
    )
    print("prediction",predictions.shape)
    print("tar_real" ,tar_real.shape)
    loss = loss_function(tar_real, predictions, target_vocab_size)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_loss.append(loss.item())

def train(transformer,decoder_vocab_size, optimizer, scheduler):
    dataset, val_input, val_output = read_data(tokenizer)

    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        start = time.time()
        train_loss.clear()
        for batch, row in enumerate(dataset):
            inp_input_ids,inp_token_type_ids,inp_attention_mask, tar_input_ids = row['inp_input_ids'].to(device), row['inp_token_type_ids'].to(device), row['inp_attention_mask'].to(device), row['tar_input_ids'].to(device)
            train_step( inp_input_ids.to(device), inp_token_type_ids.to(device), inp_attention_mask.to(device), tar_input_ids.to(device) ,decoder_vocab_size,  transformer, optimizer, scheduler)
            if batch > 0 and batch % 1000 == 0:
                print('Batch {} Loss {:.4f}'.format(batch, sum(train_loss) / len(train_loss)))
        print('Epoch {} Loss {:.4f}'.format(epoch, sum(train_loss) / len(train_loss)))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        if (epoch > 0 and epoch % 15 == 0):
            torch.save(transformer.state_dict(), f"checkpoints/transformer_epoch_{epoch}.pt")
    return val_input, val_output

if __name__ == "__main__":

    encoder_vocab_size = len(tokenizer.get_vocab().keys())
    decoder_vocab_size = len(tokenizer.get_vocab().keys())
    transformer = TransformerModel(
        num_layers, 
        d_model, 
        num_heads, 
        dff,
        decoder_vocab_size, 
        pe_target=decoder_vocab_size,
        bert=model
    ).to(device)

    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    learning_rate = CustomSchedule(optimizer, d_model)
    logging.info(f"learning rate - {learning_rate}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/10)
    

    val_input, val_output = train(transformer, decoder_vocab_size,  optimizer=optimizer, scheduler=scheduler)
    for input_document in val_input:
        result_beam = evaluate_beam(input_document, 3, 3, transformer)
        
        for e in result_beam:
            sentence_result = []
            for i in e:
                for s in i[1]:
                    if isinstance(s, str):
                        sentence_result.append(s)
                    else:
                        sentence_result.append(tokenizer.decode(s.numpy()))
            print(sentence_result)