import torch
import os, re
import json

model_files=[f for f in os.listdir('experiment') if 'model' in f]
model_files.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))

model=torch.load(os.path.join('experiment',model_files[-1]))

pre_vocab_ja=json.load(open('data/vocab/ja_vocabs.json','r',encoding='utf-8'))
pre_vocab_zh=json.load(open('data/vocab/zh_vocabs.json','r',encoding='utf-8'))
pre_ja_dict={idx+259:char for idx, char in enumerate(pre_vocab_ja)}
pre_zh_dict={idx+259:char for idx, char in enumerate(pre_vocab_zh)}
cur_vocab_ja=json.load(open('data/ja_vocabs.json','r',encoding='utf-8'))
cur_vocab_zh=json.load(open('data/zh_vocabs.json','r',encoding='utf-8'))
cur_ja_dict={char:idx+259 for idx, char in enumerate(cur_vocab_ja)}
cur_zh_dict={char:idx+259 for idx, char in enumerate(cur_vocab_zh)}

new_src_embed=torch.zeros((len(cur_vocab_ja)+259,512))
new_tgt_embed=torch.zeros((len(cur_vocab_zh)+259,512))
new_gen=torch.zeros((len(cur_vocab_zh)+259,512))
new_gen_bias=torch.zeros((len(cur_vocab_zh)+259,))
new_src_embed[:259]=model['src_embed.0.lut.weight'][:259]
new_tgt_embed[:259]=model['tgt_embed.0.lut.weight'][:259]
new_gen[:259]=model['generator.proj.weight'][:259]
new_gen_bias[:259]=model['generator.proj.bias'][:259]
for idx in range(259,len(pre_vocab_ja)+259):
    if pre_ja_dict[idx] in cur_ja_dict:
        new_src_embed[cur_ja_dict[pre_ja_dict[idx]]]=model['src_embed.0.lut.weight'][idx]
for idx in range(259,len(pre_vocab_zh)+259):
    if pre_zh_dict[idx] in cur_zh_dict:
        new_tgt_embed[cur_zh_dict[pre_zh_dict[idx]]]=model['tgt_embed.0.lut.weight'][idx]
        new_gen[cur_zh_dict[pre_zh_dict[idx]]]=model['generator.proj.weight'][idx]
        new_gen_bias[cur_zh_dict[pre_zh_dict[idx]]]=model['generator.proj.bias'][idx]
model['src_embed.0.lut.weight']=new_src_embed
model['tgt_embed.0.lut.weight']=new_tgt_embed
model['generator.proj.weight']=new_gen
model['generator.proj.bias']=new_gen_bias

torch.save(model,os.path.join('experiment',model_files[-1]))
