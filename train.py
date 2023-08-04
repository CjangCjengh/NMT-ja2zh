import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm
from torch.utils.data import DataLoader

import os, re
import config
from data_loader import MTDataset, built_dataset
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import target_tokenizer_load


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(model, model_par, criterion, optimizer):
    total_steps = 0
    for epoch in range(config.epoch_num):
        # 数据集构建
        built_dataset(config.xml_folder, config.train_data_path, config.dev_data_path, config.max_len)
        train_dataset = MTDataset(config.train_data_path)
        dev_dataset = MTDataset(config.dev_data_path)
        train_data = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                    collate_fn=train_dataset.collate_fn)
        dev_data = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                    collate_fn=dev_dataset.collate_fn)
        logging.info("-------- Dataset Build! --------")
        # 模型训练
        model.train()
        loss_compute = MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer)
        total_tokens = 0.
        total_loss = 0.
        for batch in tqdm(train_data):
            out = model_par(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens

            total_steps += 1
            if total_steps % config.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(config.model_dir, f'model_{total_steps}.pth'))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.model_dir, f'optimizer_{total_steps}.pth'))
                if os.path.exists(os.path.join(config.model_dir, f'model_{total_steps - config.save_interval*2}.pth')):
                    os.remove(os.path.join(config.model_dir, f'model_{total_steps - config.save_interval*2}.pth'))
                if os.path.exists(os.path.join(config.model_dir, f'optimizer_{total_steps - config.save_interval*2}.pth')):
                    os.remove(os.path.join(config.model_dir, f'optimizer_{total_steps - config.save_interval*2}.pth'))
                logging.info("Step: {}, Loss: {}".format(total_steps, total_loss / total_tokens))
        train_loss = total_loss / total_tokens
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # if bleu_score > best_bleu_score:
        #     best_bleu_score = bleu_score
        #     early_stop = config.early_stop
        # else:
        #     early_stop -= 1
        #     logging.info("Early Stop Left: {}".format(early_stop))
        # if early_stop == 0:
        #     logging.info("-------- Early Stop! --------")
        #     break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    _, tgt_decode = target_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            tgt_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [tgt_decode(_s) for _s in decode_result]
            trg.extend(tgt_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(lastest_checkpoint()))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    _, tgt_decode = target_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(lastest_checkpoint()))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [tgt_decode(_s) for _s in decode_result]
        print(translation[0])


def lastest_checkpoint():
    files = [f for f in os.listdir(config.model_dir) if 'model' in f]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    if len(files) == 0:
        return None
    return os.path.join(config.model_dir, files[-1])


def lastest_optimizer():
    files = [f for f in os.listdir(config.model_dir) if 'optimizer' in f]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    if len(files) == 0:
        return None
    return os.path.join(config.model_dir, files[-1])
