import os
import shutil

import numpy as np
from absl import logging

import trax
from trax.supervised import training

from . flags import FLAGS
from . data import iter_dataset, scan_for_images
from . import tokens

def generate_sample_images(training_loop, batch_itr, model):
    inp = next(batch_itr)[0]
    logits = model(inp)
    toks = np.argmax(logits, axis=-1)
    toks = np.pad(toks, ((0, 0), (1, 0)))[:, :-1]

    with training_loop._open_summary_writers() as (stl, sel):
        images = [tokens.tokens_to_image_array(toks) for toks in toks]
        images = (np.array(images) * 0xFF).astype(np.uint8)
        sel[0].images(f"gen/{training_loop._step}", images=images, step=training_loop._step, rows=2)

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
    path_images = FLAGS.images
    path_images_val = FLAGS.val_images

    # create the training and development dataset
    vocab_size = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    # XXX: switch model type on flags
    #model = trax.models.TransformerLM(vocab_size, max_len=max_length)
    model = trax.models.ReformerLM(vocab_size, max_len=max_length)

    if not (path_images_val and os.path.exists(path_images_val)):
        msg = "Warning: no validation path provided, using training images as a substitute"
        path_images_val = path_images

    training_image_list = scan_for_images(path_images)
    validation_image_list = scan_for_images(path_images_val)
    train_itr = iter_dataset(training_image_list, batch_size=FLAGS.batch_size, group="train")
    eval_itr = iter_dataset(validation_image_list, batch_size=FLAGS.batch_size, group="val")

    opt = trax.optimizers.Adam()
    loss = trax.layers.CrossEntropyLoss()
    lr = trax.lr.multifactor()

    train_task = training.TrainTask(
        labeled_data=train_itr,
        loss_layer=loss,
        lr_schedule=lr,
        optimizer=opt,
        n_steps_per_checkpoint=steps_per_epoch,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_itr,
        metrics=[trax.layers.CrossEntropyLoss(), trax.layers.Accuracy()],
        n_eval_batches=steps_per_epoch // 10
    )

    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )

    training_loop.run(1)
    if FLAGS.checkpoint:
        print(f"Loading weights from {FLAGS.checkpoint}")
        example = np.zeros([batch_size, max_length]).astype(np.int32)
        signature = trax.shapes.signature(example)
        model.init_from_file(FLAGS.checkpoint, weights_only=True, input_signature=signature)

    generate_sample_images(training_loop, eval_itr, model)
    for epoch in range(n_epochs):
        training_loop.run(steps_per_epoch)
        backup_checkpoint(output_dir, training_loop)
        generate_sample_images(training_loop, eval_itr, model)
