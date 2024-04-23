from loadData import read_data 
from config import *
import joblib
import time
from transform import TransformerModel 
from helper import *
from log_manager import *
from loadData import MyDataset
from transformers import AutoTokenizer
from transformers import BertModel
import torch
import math

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = BertModel.from_pretrained("vinai/phobert-base-v2")

lda_model = joblib.load('../input/summarize-dataset/lda_model.jl')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CosineDecayWithWarmUpScheduler(object):
    def __init__(self,optimizer,step_per_epoch,init_warmup_lr=1e-5,warm_up_steps=1000,max_lr=1e-4,min_lr=1e-6,num_step_down=2000,num_step_up=None,
                T_mul=1,max_lr_decay=None, gamma=1,min_lr_decay=None,alpha=1):
        self.optimizer = optimizer
        self.step_per_epoch = step_per_epoch
        if warm_up_steps != 0:
            self.warm_up = True
        else:
            self.warm_up = False  
        self.init_warmup_lr = init_warmup_lr
        self.warm_up_steps = warm_up_steps
        self.max_lr = max_lr
        if min_lr == 0:
            self.min_lr = 0.1 * max_lr
            self.alpha = 0.1
        else:
            self.min_lr = min_lr
        self.num_step_down = num_step_down
        if num_step_up == None:
            self.num_step_up = num_step_down
        else:
            self.num_step_up = num_step_up    
        self.T_mul = T_mul
        if max_lr_decay == None:
            self.gamma = 1
        elif max_lr_decay == 'Half':
            self.gamma = 0.5
        elif max_lr_decay == 'Exp':
            self.gamma = gamma
        
        if min_lr_decay == None:
            self.alpha = 1
        elif min_lr_decay == 'Half':
            self.alpha = 0.5
        elif min_lr_decay == 'Exp':
            self.alpha = alpha


        self.num_T = 0
        self.iters = 0
        self.lr_list = []
        
        
    def update_cycle(self, lr):
        old_min_lr = self.min_lr
        if lr == self.max_lr or (self.num_step_up == 0 and lr == self.min_lr):
            if self.num_T == 0:
                self.warm_up = False
                self.min_lr /= self.alpha
            self.iters = 0
            self.num_T += 1
            self.min_lr *= self.alpha

        if lr == old_min_lr and self.max_lr * self.gamma >= self.min_lr:
            self.max_lr *= self.gamma
            
    
    def step(self):
        self.iters += 1
        if self.warm_up:
            lr = self.init_warmup_lr + (self.max_lr-self.init_warmup_lr) / self.warm_up_steps * self.iters
        else:
            T_cur = self.T_mul**self.num_T
            if self.iters <= self.num_step_down*T_cur:
                lr = self.min_lr + (self.max_lr-self.min_lr) * (1 + math.cos(math.pi*self.iters/(self.num_step_down*T_cur)))/2
                if lr < self.min_lr:
                    lr = self.min_lr
            elif self.iters > self.num_step_down*T_cur:
                lr = self.min_lr + (self.max_lr-self.min_lr) / (self.num_step_up * T_cur) * (self.iters-self.num_step_down*T_cur)
                if lr > self.max_lr:
                    lr = self.max_lr

        self.update_cycle(lr)
                
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self.lr_list.append(lr)

# class BeamSearchNode(object):
#     def __init__(self, prev_node, token_id, log_prob):
#         self.finished = False   # Determine if the hypothesis decoding is finished
#         self.prev_node = prev_node
#         self.token_id = token_id
#         self.log_prob = log_prob

#         if prev_node is None:
#             self.seq_tokens = [token_id]
#         else:
#             self.seq_tokens = prev_node.seq_tokens + [token_id]

#         self.seq_len = len(self.seq_tokens)

#         if token_id == tokenizer.eos_token_id:
#             self.finished = True

#     def eval(self):
#         alpha = 1.0
#         reward = 0

#         # Add here a function for shaping a reward
#         return self.log_prob / float(self.seq_len - 1 + 1e-6) + alpha * reward

# def evaluate_beam(input_document, n_best, k_beam, transformer):
#     input_document = [tokenizer.encode(d, add_special_tokens=True) for d in input_document]
#     input_document = torch.nn.utils.rnn.pad_sequence([torch.tensor(d) for d in input_document], batch_first=True, padding_value=tokenizer.pad_token_id)
#     input_document = input_document.to(device)
#     encoder_input = input_document[0].unsqueeze(0)
#     decoder_input = torch.tensor(tokenizer.encode(tokenizer.special_tokens_map['cls_token'], add_special_tokens=True)).unsqueeze(0).to(device)
#     output = decoder_input.unsqueeze(0)
#     decoded_batch = []
#     beam_hypotheses = []
#     start_node = BeamSearchNode(prev_node=None, token_id=tokenizer.special_tokens_map['cls_token'], log_prob=0)
#     beam_hypotheses.append((-start_node.eval(), start_node))
#     end_nodes = []

#     for i in range(decoder_maxlen):
#         candidates = PriorityQueue()
#         for score, node in beam_hypotheses:
#             dec_seq = node.seq_tokens
#             enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
#             predictions, attention_weights, _ = transformer(
#                 encoder_input, 
#                 output,
#                 False,
#                 enc_padding_mask,
#                 combined_mask,
#                 dec_padding_mask
#             )
#             predictions = predictions[: ,-1:, :]
#             sorted_probs = torch.sort(predictions[0][0], dim=-1, descending=True)[0]
#             sorted_indices = torch.sort(predictions[0][0], dim=-1, descending=True)[1]
#             for i in range(n_best):
#                 decoded_token = sorted_indices[i]
#                 log_prob = sorted_probs[i]
        
#                 next_node = BeamSearchNode(prev_node=node, token_id=decoded_token, log_prob=node.log_prob + log_prob)
                
#                 if decoded_token == tokenizer.eos_token_id:
#                     end_nodes.append((-next_node.eval(), next_node))
#                 else:
#                     candidates.put((-next_node.eval(), next_node))
#             if len(end_nodes) >= k_beam:
#                 break
#             beam_hypotheses = [candidates.get() for _ in range(k_beam)]
#     best_hypotheses = []
#     sorted_beam_hypotheses = sorted(beam_hypotheses, key=lambda x: x[0])
    
#     if len(end_nodes) < k_beam:
#         for i in range(k_beam - len(end_nodes)):
#             end_nodes.append(sorted_beam_hypotheses[i])
#     sorted_end_nodes = sorted(end_nodes, key=lambda x: x[0])

#     for i in range(k_beam):
#         score, end_node = sorted_end_nodes[i]
#         best_hypotheses.append((score, end_node.seq_tokens))
#     decoded_batch.append(best_hypotheses)
#     return decoded_batch

import torch
from tqdm.auto import tqdm
def beam_search(
    model, 
    input_document,
    target_document,
    lda_model,
    predictions = 20,
    beam_width = 5,
    batch_size = 10, 
    progress_bar = 0
):
    with torch.no_grad():
        X = input_document["input_ids"]
        Y = torch.ones(target_document["input_ids"].shape[0], 1).to(next(model.parameters()).device).long()
        
#         Y = target_document["input_ids"]
        inp_input_ids = input_document["input_ids"]
        inp_token_type_ids = input_document["token_type_ids"]
        inp_attention_mask = input_document["attention_mask"]
        tar_input_ids = target_document["input_ids"]
        target_vocab_size = len(tokenizer.get_vocab().keys())
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(X, Y)
        pre, enc_output, att_weights = model(
            inp_input_ids, inp_token_type_ids, inp_attention_mask, tar_input_ids, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask,
            target_vocab_size,
            lda_model
            )
        next_probabilities = pre[:, -1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)
        Y = Y.repeat((beam_width, 1))
        next_chars = next_chars.reshape(-1, 1)
        Y = torch.cat((Y, next_chars), axis = -1)
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            dataset = MyDataset(input_document, target_document)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#             dataset = tud.TensorDataset(X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), Y)
#             loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(dataloader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for x in iterator:
                x["inp_input_ids"] = x["inp_input_ids"].repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1)
                x["inp_token_type_ids"] = x["inp_token_type_ids"].repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1)
                x["inp_attention_mask"] = x["inp_attention_mask"].repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1)
                x["tar_input_ids"]= x["tar_input_ids"].repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1)
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x["inp_input_ids"], x["tar_input_ids"])
                pre, enc_output, att_weights = model(
                x["inp_input_ids"], x["inp_token_type_ids"], x["inp_attention_mask"], x["tar_input_ids"], 
                enc_padding_mask, 
                combined_mask, 
                dec_padding_mask,
                target_vocab_size,
                lda_model
                )
                next_probabilities.append(pre[:, -1, :].log_softmax(-1))
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(Y.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
            Y = Y[best_candidates].flatten(end_dim = -2)
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities

def train_step(inp_input_ids, inp_token_type_ids, inp_attention_mask, tar_input_ids, tar_attention_mask, target_vocab_size, transformer, optimizer, scheduler):
    tar_real = tar_input_ids
    
    optimizer.zero_grad()

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp_input_ids, tar_input_ids)
    predictions, enc_output, att_weights = transformer(
        inp_input_ids, inp_token_type_ids, inp_attention_mask, tar_input_ids, 
        enc_padding_mask, 
        combined_mask, 
        dec_padding_mask,
        target_vocab_size,
        lda_model
    )
    desired_output = [tokenizer.decode(k) for k in tar_real]
    predict_output = [tokenizer.batch_decode(k) for k in predictions]
    predict_output = [x[:len(o)] for x, o in zip(predict_output,desired_output)]

    intersection = 0
    for idx, item in enumerate(predict_output):
      if item == desired_output[idx]:
        intersection += 1
    precision = intersection/len(desired_output)
    recall = intersection/len(predict_output)

    loss = loss_function(tar_real[:,1:,], tar_attention_mask, predictions, target_vocab_size)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item(), precision, recall

def train(transformer,decoder_vocab_size, optimizer):
    dataset, val_input, val_output = read_data(tokenizer, link_training_kaggle, link_target_kaggle)
    train_loss = []
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        start = time.time()
        train_loss.clear()
        # scheduler.step()
        scheduler = CosineDecayWithWarmUpScheduler(optimizer,step_per_epoch=1000,init_warmup_lr=1e-4,warm_up_steps=5000,max_lr=4e-4,min_lr=4e-5,num_step_down=6000,
                                          num_step_up = 6000,T_mul=1,max_lr_decay='Half')
        
        for batch, row in enumerate(dataset):
            inp_input_ids,inp_token_type_ids,inp_attention_mask, tar_input_ids, tar_attention_mask = row['inp_input_ids'].to(device), row['inp_token_type_ids'].to(device), row['inp_attention_mask'].to(device), row['tar_input_ids'].to(device), row['tar_attention_mask'].to(device)
            loss, precision, recall = train_step( inp_input_ids.to(device), inp_token_type_ids.to(device), inp_attention_mask.to(device), tar_input_ids.to(device), tar_attention_mask.to(device) ,decoder_vocab_size,  transformer, optimizer, scheduler)
            train_loss.append(loss)
            if batch > 0 and batch % 500 == 0:
                print('Batch {} Loss {:.4f}'.format(batch, sum(train_loss) / len(train_loss)))
        print('Epoch {} Loss {:.4f} Precision {:4f} Recall {:4f}'.format(epoch, sum(train_loss) / len(train_loss), precision, recall))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        if (epoch > 0 and epoch % 5 == 0):
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

    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    # learning_rate = CustomSchedule(optimizer, d_model)
    # logging.info(f"learning rate - {learning_rate}")
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/10)
    

    val_input, val_output = train(transformer, decoder_vocab_size,  optimizer=optimizer)
    
    model_test = TransformerModel(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    decoder_vocab_size, 
    pe_target=decoder_vocab_size,
    bert=model
    )
    model_test = transformer
    result_beam = beam_search(model_test.to(device), val_input.to(device), val_output.to(device),lda_model)
    predict_sum = [tokenizer.batch_decode(k) for k in result_beam[0]]
    print(predict_sum)
    # for input_document in val_input:
    #     result_beam = evaluate_beam(input_document, 3, 3, transformer)
        
    #     for e in result_beam:
    #         sentence_result = []
    #         for i in e:
    #             for s in i[1]:
    #                 if isinstance(s, str):
    #                     sentence_result.append(s)
    #                 else:
    #                     sentence_result.append(tokenizer.decode(s.numpy()))
    #         print(sentence_result)