import os
import time
import random
import multiprocessing as mp

from PIL import Image
import numpy as np

from absl import logging
from . import tokens
from . flags import FLAGS

def _shuffle_forever(item_list):
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
    def _task_fill_image(self, toks):
        n_toks = len(toks)
        fill = tokens.special_token("<fill>")
        cut = random.randint(1, n_toks - 1)
        x = toks[:cut] + ([fill] * (n_toks - cut))
        w = ([0] * n_toks)
        w[cut] = 1
        y = toks[:]

        x = np.array(x).astype(np.int32)
        y = np.array(y).astype(np.int32)
        w = np.array(w).astype(np.float)

        return (x, y, w)

    def _task_next_token(self, toks):
        pad = tokens.special_token("<pad>")
        x = np.array(toks[:-1] + [pad]).astype(np.int32)
        y = np.array(toks[1:] + [pad]).astype(np.int32)
        w = ([1] * (len(x) - 1))  + [0]
        w = np.array(w).astype(np.float)
        return (x, y, w)

    def _auto_regress(self, work):
        work = np.array(work).astype(np.int32)
        mask = np.ones_like(work, dtype=np.float)
        return (work, work, mask)

    def _do_work(self, img_name):
        img_path = self._work_list[img_name]
        img = Image.open(img_path)
        work = tokens.image_to_tokens(img, size=FLAGS.image_size)
        work = list(work)
        #work = self._auto_regress(work)
        work = self._task_fill_image(work)
        assert len(work) == 3
        return work

def _deque(que, qcon):
    with qcon:
        while que.empty():
            print(f"_deque(): queue empty")
            qcon.wait(1)
        item = que.get()
        qcon.notify()
    return item

def _gather(que, qcon):
    while True:
        yield _deque(que, qcon)

def iter_debug(batch_size=None):
    max_length = FLAGS.image_size ** 2
    ary = np.ones((batch_size, max_length))
    while True:
        yield (ary.astype(np.int32), ary.astype(np.int32), ary)

def iter_dataset(work_list, batch_size=None, n_workers=4, qsize=1024, group=None):
    group = group or "default"
    que = mp.Queue(qsize)
    qcon = mp.Condition()
    batch_size = batch_size or FLAGS.batch_size

    workers = []
    print(f"Starting {n_workers} workers for group {group}")
    for n in range(n_workers):
        worker = ImageWorker(work_list, batch_size, que, qcon)
        worker.start()
        workers.append(worker)

    return _gather(que, qcon)
