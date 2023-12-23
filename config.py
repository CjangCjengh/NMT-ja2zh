import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 1
eos_idx = 2
src_vocab_size = 6760
tgt_vocab_size = 5759
batch_size = 12
epoch_num = 100
save_interval = 1000
early_stop = 5
lr = 3e-4

# greed decode的最大句子长度
max_len = 900
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

src_vocab_path = './data/vocab/src_vocabs.json'
tgt_vocab_path = './data/vocab/zh_vocabs.json'

data_dir = './data'
xml_folder = './data/text/xml'
name_list_path = './data/text/name_list.json'
xml_template_folder = './data/text/xml_template'
name_folder = './data/text/ko_name'
train_data_path = './data/text/train.json'
dev_data_path = './data/text/dev.json'
test_data_path = './data/text/test.json'
model_dir = './experiment'
log_path = f'{model_dir}/train.log'
output_path = f'{model_dir}/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
