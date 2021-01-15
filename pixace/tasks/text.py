import numpy as np
from PIL import Image

from .. utils import scan_for_images
from .. tokens import ImageTokenModel
from . base import BaseTask, BatchWorker, DatasetGenerator

class SentenceWorker(BatchWorker):
    def __init__(self, tokenizer=None, **kw):
        super().__init__(**kw)
        self.tokenizer = tokenizer

    def process_job(self, text):
        work = self.tokenizer.encode(text)
        work = np.array(work).astype(np.int32)
        weights = np.ones_like(work, dtype=np.float)
        return (work, work, weights)

class SentenceTask(BaseTask):
    def __init__(self, corpus=None, max_len=None, tokenizer=None, batch_size=None, group=None, **kw): 
        super().__init__(**kw)
        self.group = group
        self.corpus = tuple(corpus)
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __iter__(self):
        dsgen = DatasetGenerator(
            data=self.shuffle_forever(self.corpus),
            group=self.group,
            worker_class=TextWorker,
            worker_config=dict(
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_len=max_len
            )
        )

        return iter(dsgen)

    def render_samples(self, logits, n_samples=None):
        if n_samples:
            logits = logits[:n_samples, ...]
        toks = np.argmax(logits, axis=-1)

        results = []
        for tk in toks:
            try:
                text = self.tokenizer.decode(tk)
                results.append(text)
            except ValueError:
                results.append("<error>")

        return {"text": results}

    @classmethod
    def build(cls, corpus=None, fn_model=None, max_len=None, batch_size=None, group=None):
        text = [ln for ln in text if len(ln) < max_len]
        if len(text) == 0:
            msg = f"No text provided"
            raise ValueError(msg)

        tokenizer = TextTokenModel(
            fn_model=fn_model,
            max_len=max_len
        )

        task = cls(
            corpus=corpus,
            max_len=max_len,
            tokenizer=tokenizer,
            batch_size=batch_size,
            group=group
        )

        return task
