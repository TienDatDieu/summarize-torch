import torch
import torch.nn as nn
from transformers import BertModel

class BertLayer(nn.Module):
    """
    Custom PyTorch module, integrating BERT from transformers
    """
    def __init__(self, bert):
        super(BertLayer, self).__init__()
        self.bert = bert

    def forward(self, inputs):
        result = self.bert(**inputs)
        return result['last_hidden_state']

# # Example usage:
# # Load a pre-trained BERT model
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# # Create an instance of the custom PyTorch module
# bert_layer = BertLayer(bert_model)

# # Assuming 'inputs' is a dictionary with keys 'input_ids', 'attention_mask', and 'token_type_ids'
# inputs = {
#     'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
#     'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
#     'token_type_ids': torch.tensor([[0, 0, 0, 0, 0]])
# }

# # Forward pass through the custom PyTorch module
# output = bert_layer(inputs)
# print(output)