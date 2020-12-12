import os
import shutil

from absl import logging

import trax
from trax.supervised import training

from . flags import FLAGS
from . globdb import GlobDatabase
from . data import iter_dataset
from . import tokens
from . layers import WeightedCategoryAccuracy, WeightedCategoryCrossEntropy

def backup_checkpoint(output_dir, training_loop):
    old_path = os.path.join(output_dir, f"model.pkl.gz")
    if not os.path.exists(old_path):
        return
    new_path = os.path.join(output_dir, f"model-{training_loop.step:05d}.pkl.gz")
    shutil.copyfile(old_path, new_path)

def train_model(argv):
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    steps_per_epoch = FLAGS.steps_per_epoch
    bitdepth = FLAGS.bitdepth
    n_epochs = FLAGS.n_epochs

    # create the training and development dataset
    vocab_size = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    model = trax.models.TransformerLM(vocab_size, max_len=max_length)

    imgdb = GlobDatabase(FLAGS.images, "*.jpg")

    work_list = imgdb.select("train")
    train_itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="train")
    work_list = imgdb.select("val")
    eval_itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="val")

    train_task = training.TrainTask(
        labeled_data=train_itr,
        loss_layer=WeightedCategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=steps_per_epoch,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_itr,
        metrics=[WeightedCategoryCrossEntropy(), WeightedCategoryAccuracy()],
        n_eval_batches=steps_per_epoch // 10
    )

    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )

    for epoch in range(n_epochs):
        training_loop.run(steps_per_epoch)
        backup_checkpoint(output_dir, training_loop)
        #lm_generate_example(training_loop, eval_batch, train_model, dc_train.captions)
