#@title
# hyper-params
num_layers = 12
d_model = 768
dff = 512
num_heads = 8
EPOCHS = 50


encoder_maxlen = 256
decoder_maxlen = 50

BUFFER_SIZE = 20000
BATCH_SIZE = 32
link_training_kaggle = '../input/summarize-dataset/train_pharagraph_full.jl'
link_target_kaggle = '../input/summarize-dataset/target_strings_full.jl'