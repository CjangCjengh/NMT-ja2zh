import os
import logging
import json


def target_tokenizer_load():
    return load_tokenizer('./data/vocab/zh_vocabs.json')


def source_tokenizer_load():
    return load_tokenizer('./data/vocab/ja_vocabs.json')


def load_tokenizer(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabs = json.load(f)
    vocab_to_id = {v:i+259 for i, v in enumerate(vocabs)}
    id_to_vocab = {v:k for k, v in vocab_to_id.items()}
    def encode(text):
        token_list = []
        for token in text:
            if token in vocab_to_id:
                token_list.append(vocab_to_id[token])
            else:
                for c in token.encode('utf-16 be'):
                    token_list.append(c+3)
        return token_list
    def decode(ids):
        text = ''
        i = 0
        while i < len(ids):
            if ids[i] >= 259:
                text += id_to_vocab[ids[i]]
                i+=1
            elif ids[i] > 2:
                char = (ids[i]-3).to_bytes(1, 'big')
                i+=1
                while i < len(ids) and ids[i] > 2 and ids[i] < 259:
                    char += (ids[i]-3).to_bytes(1, 'big')
                    i+=1
                try:
                    text += char.decode('utf-16 be')
                except:
                    continue
            else:
                i+=1
        return text
    return encode, decode


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
