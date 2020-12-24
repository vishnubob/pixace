import sys
import os
from absl import app

from . flags import FLAGS, load_flags
from . train import train_model
from . predict import predict_model
from . zoo import download_model

__all__ = ["cli", "train", "predict", "download"]

def _handle_flags(argv):
    FLAGS.bitdepth = [int(bit) for bit in FLAGS.bitdepth]
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_name)

def runner(argv):
    command = argv[1]

    if command == "train":
        _handle_flags(argv)
        train_model(argv)
    elif command == "predict":
        _handle_flags(argv)
        predict_model(argv)
    elif command == "download":
        download_model(argv)

def cli():
    command = sys.argv[1]
    load_flags(command)
    app.run(runner)

def wrap_command(command, **kw):
    argv = [sys.argv[0], command]
    argv += [f"--{key}={val}" for (key, val) in kw.items()]
    sys.argv = argv
    print(sys.argv)
    return cli()

def train(**kw):
    return wrap_command("train", **kw)

def predict(**kw):
    return wrap_command("predict", **kw)

def download(**kw):
    return wrap_command("download", **kw)
