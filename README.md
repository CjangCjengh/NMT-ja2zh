Based on [hemingkx / ChineseNMT](https://github.com/hemingkx/ChineseNMT)

## Prepare Data
参考 data/text/xml/example.xml ，一本书一个xml，训练时会随机合并上下文

## Tokenize
目前中文和日文均采取一个字一个token

## Train
```shell
python train.py
```

## Infer
```shell
python translate.py
```
