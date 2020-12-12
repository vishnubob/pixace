import os
from absl import app

from . flags import FLAGS
from . train import train_model
from . predict import predict_model

def _handle_flags(argv):
    FLAGS.bitdepth = [int(bit) for bit in FLAGS.bitdepth]
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.run_name)

def runner(argv):
    _handle_flags(argv)

    if FLAGS.mode == "train":
        train_model(argv)
    elif FLAGS.mode == "predict":
        predict_model(argv)

def cli():
    app.run(runner)
