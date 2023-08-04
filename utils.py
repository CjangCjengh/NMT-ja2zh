import os
import logging
import json


def target_tokenizer_load():
    with open('./data/vocab/zh_vocabs.json', 'r', encoding='utf-8') as f:
        zh_vocabs = json.load(f)
    vocab_to_id = {v:i+3 for i, v in enumerate(zh_vocabs)}
    id_to_vocab = {v:k for k, v in vocab_to_id.items()}
    def encode(text):
        return [vocab_to_id.get(token, 4) for token in text]
    def decode(ids):
        return ''.join([id_to_vocab.get(i, '') for i in ids])
    return encode, decode


def source_tokenizer_load():
    with open('./data/vocab/ja_vocabs.json', 'r', encoding='utf-8') as f:
        ja_vocabs = json.load(f)
    vocab_to_id = {v:i+3 for i, v in enumerate(ja_vocabs)}
    id_to_vocab = {v:k for k, v in vocab_to_id.items()}
    def encode(text):
        return [vocab_to_id.get(token, 4) for token in text]
    def decode(ids):
        return ''.join([id_to_vocab.get(i, '') for i in ids])
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


