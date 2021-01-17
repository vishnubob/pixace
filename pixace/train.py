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

def render_samples(training_loop, logits, task, rows=2):
    res = task.render_samples(logits)
    with training_loop._open_summary_writers() as (stl, sel):
        for (key, val) in res.items():
            if key == "images":
                sel[0].images(f"gen/{training_loop._step}", images=val, step=training_loop._step, rows=rows)

def backup_checkpoint(output_dir, training_loop):
    old_path = os.path.join(output_dir, f"model.pkl.gz")
    if not os.path.exists(old_path):
        return
    new_path = os.path.join(output_dir, f"model-{training_loop.step:05d}.pkl.gz")
    shutil.copyfile(old_path, new_path)

class Trainer(object):
    def __init__(self, model_name=None, model_type="reformer", weights_dir="model-weights", tokenizer=None):
        self.model_name = model_name or _get_ts()
        self.model_type = model_type
        self.weights_dir = weights_dir
        self.tokenizer = tokenizer
    
    def init_generators(self, batch_size=None, train=None, val=None):
        with open(train) as fh:
            train = json.load(fh)

        if val:
            with open(val) as fh:
                val = json.load(fh)
        else:
            msg = "Warning: no validation data, using training images as a substitute"
            val = train

        train_gen = batch_generator(train, tokenizer=self.tokenizer, batch_size=batch_size)
        val_gen = batch_generator(val, tokenizer=self.tokenizer, batch_size=batch_size)
        return (train_gen, val_gen)

    def init_model(self):
        msg = f"Initializing {self.model_type} model (n_tokens={self.n_tokens}, max_len={self.max_len}, image_size={self.image_size}, bitdepth={self.bitdepth})"
        print(msg)
        if self.model_type == "transformer":
            model = trax.models.TransformerLM(self.n_tokens, max_len=self.max_len, mode="train")
        elif self.model_type == "reformer":
            model = trax.models.ReformerLM(self.n_tokens, max_len=self.max_len, mode="train")
        else:
            msg = f"Unknown model type '{self.model_type}'"
            raise ValueError(msg)
        return model

    def train(self, batch_size=8, steps_per_epoch=100, steps_per_eval=None, n_epochs=10, train_data=None, val_data=None, max_len=None, spm_model=None):
        output_dir = os.path.join(self.weights_dir, self.model_name)
        lr = trax.lr.multifactor()
        loss = tl.WeightedCategoryCrossEntropy()
        eval_metrics = [
            tl.WeightedCategoryCrossEntropy(), 
            tl.WeightedCategoryAccuracy(),
        ]
        opt = trax.optimizers.Adam()
        if steps_per_eval is None:
            steps_per_eval = max(1, steps_per_epoch // 10)

        (train_gen, eval_gen) = self.init_generators(batch_size=batch_size, train=train_data, val=val_data)
        (train_itr, eval_itr) = (iter(train_gen), iter(eval_gen))

        self.n_tokens = train_gen.tokenizer.n_tokens

        model = self.init_model()
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
            model,
            train_task,
            eval_tasks=[eval_task],
            output_dir=output_dir
        )

        sample_batch = next(eval_itr)
        logits = model(sample_batch[0])
        render_samples(training_loop, logits, eval_gen)

        for epoch in range(n_epochs):
            training_loop.run(steps_per_epoch)
            backup_checkpoint(output_dir, training_loop)
            
            # sample output
            sample_batch = next(eval_itr)
            logits = model(sample_batch[0])
            render_samples(training_loop, logits, eval_gen)

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS

        trainer = cls(
            model_name=FLAGS.model_name,
            model_type=FLAGS.model_type,
            weights_dir=FLAGS.weights_dir,
            tokenizer=FLAGS.tokenizer,
        )

        trainer.train(
            train_data=FLAGS.train_data,
            val_data=FLAGS.val_data,
            batch_size=FLAGS.batch_size,
            steps_per_epoch=FLAGS.steps_per_epoch,
            steps_per_eval=FLAGS.steps_per_eval,
            n_epochs=FLAGS.n_epochs,
        )
