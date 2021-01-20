import os
import json
import time
import shutil

import numpy as np
from absl import logging

import trax
import trax.layers as tl
from trax.supervised import training

from . task import batch_generator

_get_ts = lambda: time.strftime("%m%d_%H%M")

def render_samples(training_loop, logits, tokenizer, rows=2):
    toks = tl.logsoftmax_sample(logits)
    toks = np.array(toks)
    res = {}
    images = []
    for sample in toks:
        sample = tokenizer.decode(sample)
        for key in sample:
            if key not in res:
                res[key] = list()
            res[key].append(sample[key])
            # XXX: hard wired key, should be based on type
            if key == "image":
                images.append(np.array(sample[key], dtype=np.uint8))

    with training_loop._open_summary_writers() as (stl, sel):
        #for (key, val) in res.items():
            #if key == "image":
                #sel[0].images(f"gen/{training_loop._step}", images=val, step=training_loop._step, rows=rows)
        sel[0].images(f"gen/{training_loop._step}", images=images, step=training_loop._step, rows=rows)

def backup_checkpoint(output_dir, training_loop):
    old_path = os.path.join(output_dir, f"model.pkl.gz")
    if not os.path.exists(old_path):
        return
    new_path = os.path.join(output_dir, f"model-{training_loop.step:05d}.pkl.gz")
    shutil.copyfile(old_path, new_path)

class Trainer(object):
    def __init__(self, model=None, tokenizer=None, output_dir=None):
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
    
    def init_generators(self, batch_size=None, train=None, val=None):
        if val is None:
            msg = "Warning: no validation data, using training images as a substitute"
            val = train

        train_gen = batch_generator(train, tokenizer=self.tokenizer, batch_size=batch_size, group="train")
        val_gen = batch_generator(val, tokenizer=self.tokenizer, batch_size=batch_size, group="validation")
        return (train_gen, val_gen)

    def train(self, train_data=None, val_data=None, batch_size=None, **kw):
        (train_gen, eval_gen) = self.init_generators(batch_size=batch_size, train=train_data, val=val_data)
        (train_itr, eval_itr) = (iter(train_gen), iter(eval_gen))
        try:
            self.train_core(train_itr=train_itr, eval_itr=eval_itr, batch_size=batch_size, **kw)
        finally:
            train_gen.stop()
            eval_gen.stop()

    def train_core(self, batch_size=8, steps_per_epoch=100, steps_per_eval=None, n_epochs=10, train_itr=None, eval_itr=None, output_dir=None):
        lr = trax.lr.multifactor()
        loss = tl.WeightedCategoryCrossEntropy()
        eval_metrics = [
            tl.WeightedCategoryCrossEntropy(), 
            tl.WeightedCategoryAccuracy(),
        ]
        opt = trax.optimizers.Adam()
        if steps_per_eval is None:
            steps_per_eval = max(1, steps_per_epoch // 10)

        train_task = training.TrainTask(
            labeled_data=train_itr,
            loss_layer=loss,
            lr_schedule=lr,
            optimizer=opt,
            n_steps_per_checkpoint=steps_per_epoch,
        )

        eval_task = training.EvalTask(
            labeled_data=eval_itr,
            metrics=eval_metrics,
            n_eval_batches=steps_per_eval
        )

        training_loop = training.Loop(
            self.model,
            train_task,
            eval_tasks=[eval_task],
            output_dir=output_dir
        )

        sample_batch = next(eval_itr)
        logits = self.model(sample_batch[0])
        render_samples(training_loop, logits, self.tokenizer)

        for epoch in range(n_epochs):
            training_loop.run(steps_per_epoch)
            backup_checkpoint(output_dir, training_loop)
            
            # sample output
            sample_batch = next(eval_itr)
            logits = self.model(sample_batch[0])
            render_samples(training_loop, logits, self.tokenizer)

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS
        from . factory import get_factory

        factory = get_factory()
        (model, tokenizer) = factory.init_model(mode="train")
        trainer = cls(model=model, tokenizer=tokenizer)

        trainer.train(
            train_data=FLAGS.train_data,
            val_data=FLAGS.val_data,
            batch_size=FLAGS.batch_size,
            steps_per_epoch=FLAGS.steps_per_epoch,
            steps_per_eval=FLAGS.steps_per_eval,
            n_epochs=FLAGS.n_epochs,
            output_dir=factory.output_dir
        )
