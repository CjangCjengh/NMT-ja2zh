import torch
import json
import numpy as np
import os, re
import random
import glob
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import source_tokenizer_load
from utils import target_tokenizer_load

import config
DEVICE = config.device


def built_dataset(xml_folder, train_data_path, dev_data_path, max_length, prob=0.85):
    train_data=[]
    xml_files=glob.glob(f'{xml_folder}/**/*.xml', recursive=True)
    for file in xml_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
            src_lines = re.findall(r'<ja>(.*?)</ja>', data)
            tgt_lines = re.findall(r'<zh>(.*?)</zh>', data)
            i=0
            while i < len(src_lines):
                if tgt_lines[i]=='':
                    i+=1
                    continue
                src_line = src_lines[i].replace('\\n', '\n')
                tgt_line = tgt_lines[i].replace('\\n', '\n')
                if len(src_line) > max_length:
                    i+=1
                    continue
                while random.random() < prob and i+1<len(src_lines) and tgt_lines[i+1]!='' and len(src_line)+len(src_lines[i+1])+1 <= max_length:
                    i+=1
                    src_line += '\n'+src_lines[i].replace('\\n', '\n')
                    tgt_line += '\n'+tgt_lines[i].replace('\\n', '\n')
                train_data.append([src_line, tgt_line])
                i+=1

    random.shuffle(train_data)
    dev_data = train_data[-100:]
    train_data = train_data[:-100]

    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_src_sent, self.out_tgt_sent = self.get_dataset(data_path, sort=True)
        self.sp_src = source_tokenizer_load()[0]
        self.sp_tgt = target_tokenizer_load()[0]
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = json.load(open(data_path, 'r'))
        out_src_sent = []
        out_tgt_sent = []
        for idx in range(len(dataset)):
            out_src_sent.append(dataset[idx][0])
            out_tgt_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_src_sent)
            out_src_sent = [out_src_sent[i] for i in sorted_index]
            out_tgt_sent = [out_tgt_sent[i] for i in sorted_index]
        return out_src_sent, out_tgt_sent

    def __getitem__(self, idx):
        eng_text = self.out_src_sent[idx]
        chn_text = self.out_tgt_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_src_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_src(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_tgt(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
