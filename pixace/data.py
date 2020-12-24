import os
import time
import random
from pathlib import Path
from fnmatch import fnmatch
import multiprocessing as mp

from PIL import Image
import numpy as np

from absl import logging
from . import tokens

def _shuffle_forever(item_list):
    # XXX: use jax rng
    while True:
        item_list = list(item_list)
        random.shuffle(item_list)
        for item in item_list:
            yield item

class ListWorker(mp.Process):
    def __init__(self, work_list=None, batch_size=None, que=None, qcon=None):
        super().__init__()
        self.daemon = True
        self._work_list = work_list
        self._batch_size = batch_size
        self._que = que
        self._qcon = qcon
        self._running = False

    def _enque(self, item):
        with self._qcon:
            while self._que.full():
                self._qcon.wait(1)
                if not self._running:
                    return
            self._que.put(item)
            self._qcon.notify()

    def _do_work(self, item):
        pass

    def _get_batch(self, itr):
        assert self._batch_size > 0
        # get list of work
        batch = [next(itr) for idx in range(self._batch_size)]
        batch = list(map(self._do_work, batch))
        batch = [np.vstack(it) for it in zip(*batch)]
        return batch

    def run(self):
        self._running = True
        work_gen = _shuffle_forever(self._work_list)
        while self._running:
            batch = self._get_batch(work_gen)
            self._enque(batch)

class ImageWorker(ListWorker):
    def __init__(self, image_size=None, bitdepth=None, **kw):
        super().__init__(**kw)
        self.image_size = image_size
        self.bitdepth = bitdepth

    def _auto_regress(self, work):
        work = np.array(work).astype(np.int32)
        mask = np.ones_like(work, dtype=np.float)
        return (work, work, mask)

    def _do_work(self, img_name):
        img_path = self._work_list[img_name]
        img = Image.open(img_path)
        work = tokens.image_to_tokens(img, size=self.image_size, bitdepth=self.bitdepth)
        work = list(work)
        work = self._auto_regress(work)
        assert len(work) == 3
        return work

def _deque(que, qcon):
    with qcon:
        while que.empty():
            msg = f"[warning] queue empty, maybe increase worker count?"
            print(msg)
            qcon.wait(1)
        item = que.get()
        qcon.notify()
    return item

def _gather(que, qcon):
    while True:
        yield _deque(que, qcon)

def iter_dataset(work_list, batch_size=None, image_size=None, bitdepth=None, n_workers=4, qsize=1024, group=None):
    group = group or "default"
    que = mp.Queue(qsize)
    qcon = mp.Condition()
    print(f"Starting {n_workers} workers within the '{group}' group")

    workers = []
    for n in range(n_workers):
        worker = ImageWorker(work_list=work_list, batch_size=batch_size, image_size=image_size, bitdepth=bitdepth, que=que, qcon=qcon)
        worker.start()
        workers.append(worker)

    # let workers settle
    time.sleep(1)

    return _gather(que, qcon)

def scan_for_images(path, patterns=None):
    patterns = patterns or ["*.jpg", "*.jpeg", "*.png"]

    _db = {}
    match_img = lambda fn: any([fnmatch(fn.lower(), pat) for pat in patterns])

    for (root, dirs, files) in os.walk(path):
        root = Path(root)
        for fn in files:
            if not match_img(fn):
                continue
            pt = root.joinpath(fn)
            assert pt.stem not in _db
            _db[pt.stem] = pt

    if not len(_db):
        msg = f"No images under '{path}' were found matching '{patterns}'"
        raise ValueError(msg)

    msg = f"Found {len(_db)} images under '{path}'"
    print(msg)
    return _db
