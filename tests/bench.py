import os
import json
import tempfile
import numpy as np
from PIL import Image
import random
import pixace
from pixace import tokens
import pytest

def bench(with_corpus=False, seed=0):
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

        def make_test_image(tdir):
            img = np.ones((32, 32, 3), dtype=np.uint8) * 255
            img = Image.fromarray(img)
            imgfn = os.path.join(tdir, "test.png")
            img.save(imgfn)

        def make_test_input(tdir):
            #res = [{"image": "test.png", "text": "text"}]
            imgfn = os.path.join(tdir, "test.png")
            assert os.path.exists(imgfn)
            res = [{"image": imgfn}]
            with open("test.json", "w") as fh:
                json.dump(res, fh)

        def _bench():
            random.seed(seed)
            args = []
            with tempfile.TemporaryDirectory() as tdir:
                cwd = os.getcwd()
                os.chdir(tdir)
                make_test_image(tdir)
                make_test_input(tdir)
                if with_corpus:
                    corpus = make_test_corpus()
                    args += [corpus]
                try:
                    res = func(*args)
                finally:
                    os.chdir(cwd)
            return res
        return _bench
    return decorator
