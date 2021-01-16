import os
import tempfile
import logging

import numpy as np
from PIL import Image

import pytest
import random
from pixace import tokens

logger = logging.getLogger()

def bench(with_corpus=False):
    def decorator(func):
        def rand_and_join(lst, minmax, joinc):
            random.shuffle(lst)
            wlen = random.randint(*minmax)
            return str.join(joinc, lst[:wlen])

        def make_test_corpus():
            alpha = [chr(x) for x in range(ord('a'), ord('z') + 1)]
            words = [rand_and_join(alpha, (1, 10), '') for x in range(1000)]
            corpus = [rand_and_join(words, (5, 200), ' ') for x in range(1000)]
            with open("corpus.txt", "w") as fh:
                fh.write(str.join('\n', corpus) + '\n')
            return corpus

        def make_test_image():
            img = np.ones((32, 32, 3), dtype=np.uint8) * 255
            img = Image.fromarray(img)
            img.save("test.png")

        def _bench():
            random.seed(0)
            args = []
            with tempfile.TemporaryDirectory() as tdir:
                cwd = os.getcwd()
                os.chdir(tdir)
                make_test_image()
                if with_corpus:
                    corpus = make_test_corpus()
                    args += [corpus]
                try:
                    return func(*args)
                finally:
                    os.chdir(cwd)
        return _bench
    return decorator

@bench()
def test_image_tokenizer():
    im = tokens.ImageTokenModel(image_size=(32, 32), bitdepth=(1, 1, 1))
    img = Image.open("test.png")
    e_img = im.encode_image(img)
    assert len(e_img) == (32 * 32 + 2)
    d_img = im.decode_image(e_img)
    assert np.all(np.array(d_img) == np.array(img))

@bench(with_corpus=True)
def test_text_tokenizer(corpus):
    tm = tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000)
    text = corpus[0]
    enc_text = tm.encode(text)
    dec_text = tm.decode(enc_text)
    assert text == dec_text

@bench(with_corpus=True)
def test_serial_tokenizer(corpus):
    logger.disabled = True
    tm = tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000)
    logger.disabled = False
    im = tokens.ImageTokenModel(image_size=(32, 32), bitdepth=(1, 1, 1))
    sm = tokens.SerialTokenModel(
        models={"IMAGE": im, "TEXT": tm},
        order=["IMAGE", "TEXT"]
    )

    text = corpus[0]
    img = "test.png"
    img = Image.open(img)
    item = {"TEXT": text, "IMAGE": img}
    
    encoded_item = sm.encode(item)
    decoded_item = sm.decode(encoded_item)
    assert decoded_item["TEXT"] == item["TEXT"]
    assert np.all(np.array(decoded_item["IMAGE"]) == np.array(item["IMAGE"]))

@bench(with_corpus=True)
def test_tokenizer_config(corpus):
    tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000, save_as="test.spm")

    tokenizers = [
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3',
        'type=text,key=label,model_file=test.spm,max_len=500',
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3'
    ]

    tokenizer = tokens.config.parse_and_build_tokenizer(tokenizers)
