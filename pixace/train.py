import os
import shutil

import numpy as np
from absl import logging

import trax
from trax.supervised import training

from . data import iter_dataset, scan_for_images
from . import tokens

def generate_sample_images(training_loop, batch_itr, model, image_size=None, bitdepth=None):
    inp = next(batch_itr)[0]
    logits = model(inp)
    toks = np.argmax(logits, axis=-1)
    toks = np.pad(toks, ((0, 0), (1, 0)))[:, :-1]

    with training_loop._open_summary_writers() as (stl, sel):
        images = [tokens.tokens_to_image_array(toks, bitdepth=bitdepth) for toks in toks]
        images = (np.array(images) * 0xFF).astype(np.uint8)
        sel[0].images(f"gen/{training_loop._step}", images=images, step=training_loop._step, rows=2)

def backup_checkpoint(output_dir, training_loop):
    old_path = os.path.join(output_dir, f"model.pkl.gz")
    if not os.path.exists(old_path):
        return
    new_path = os.path.join(output_dir, f"model-{training_loop.step:05d}.pkl.gz")
    shutil.copyfile(old_path, new_path)

class Trainer(object):
    def __init__(self, model_name=None, model_type="reformer", weights_dir="model-weights", image_size=32, bitdepth=(5,4,4)):
        self.model_name = model_name
        self.model_type = model_type
        self.weights_dir = weights_dir
        self.bitdepth = bitdepth
        self.image_size = image_size
        self.max_length = self.image_size ** 2
        self.n_tokens = tokens.token_count(bitdepth=self.bitdepth)
    
    def init_data(self, batch_size=None, images=None, val_images=None):
        training_image_list = scan_for_images(images)
        train_itr = iter_dataset(
            training_image_list, 
            batch_size=batch_size, 
            image_size=self.image_size, 
            bitdepth=self.bitdepth, 
            group="train"
        )

        if val_images is None:
            msg = "Warning: no validation path provided, using training images as a substitute"
            val_images = images

        val_images = scan_for_images(val_images)
        eval_itr = iter_dataset(
            val_images, 
            batch_size=batch_size, 
            image_size=self.image_size, 
            bitdepth=self.bitdepth, 
            group="val"
        )
        return (train_itr, eval_itr)

    def init_model(self):
        msg = f"Initializing {self.model_type} model (n_tokens={self.n_tokens}, max_len={self.max_length}, image_size={self.image_size}, bitdepth={self.bitdepth})"
        print(msg)
        if self.model_type == "transformer":
            model = trax.models.TransformerLM(self.n_tokens, max_len=self.max_length, mode="train")
        elif self.model_type == "reformer":
            model = trax.models.ReformerLM(self.n_tokens, max_len=self.max_length, mode="train")
        else:
            msg = f"Unknown model type '{self.model_type}'"
            raise ValueError(msg)
        return model

    def train(self, batch_size=None, steps_per_epoch=1000, steps_per_eval=None, n_epochs=100, images="images", val_images=None):
        opt = trax.optimizers.Adam()
        loss = trax.layers.CrossEntropyLoss()
        lr = trax.lr.multifactor()
        model = self.init_model()
        if steps_per_eval is None:
            steps_per_eval = max(1, steps_per_epoch // 10)

        (train_itr, eval_itr) = self.init_data(batch_size=batch_size, images=images, val_images=val_images)
        batch = next(train_itr)

        train_task = training.TrainTask(
            labeled_data=train_itr,
            loss_layer=loss,
            lr_schedule=lr,
            optimizer=opt,
            n_steps_per_checkpoint=steps_per_epoch,
        )

        eval_task = training.EvalTask(
            labeled_data=eval_itr,
            metrics=[trax.layers.CrossEntropyLoss(), trax.layers.SequenceAccuracy()],
            n_eval_batches=steps_per_eval
        )

        output_dir = os.path.join(self.weights_dir, self.model_name)
        training_loop = training.Loop(
            model,
            train_task,
            eval_tasks=[eval_task],
            output_dir=output_dir
        )

        for epoch in range(n_epochs):
            training_loop.run(steps_per_epoch)
            backup_checkpoint(output_dir, training_loop)
            generate_sample_images(training_loop, eval_itr, model, image_size=self.image_size, bitdepth=self.bitdepth)

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS

        trainer = cls(
            model_name=FLAGS.model_name,
            model_type=FLAGS.model_type,
            weights_dir=FLAGS.weights_dir,
            image_size=FLAGS.image_size,
            bitdepth=FLAGS.bitdepth,
        )

        trainer.train(
            batch_size=FLAGS.batch_size,
            steps_per_epoch=FLAGS.steps_per_epoch,
            steps_per_eval=FLAGS.steps_per_eval,
            n_epochs=FLAGS.n_epochs,
            images=FLAGS.images,
            val_images=FLAGS.val_images
        )
