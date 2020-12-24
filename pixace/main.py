import sys
import os
from absl import app

from . flags import FLAGS, load_flags, reset_flags
from . train import train_model
from . inference import predict_model
from . zoo import download_model

def _handle_flags(argv):
    FLAGS.bitdepth = [int(bit) for bit in FLAGS.bitdepth]
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_name)

def runner(argv):
    command = argv[1]

    if command == "train":
        _handle_flags(argv)
        return train_model(argv)
    elif command == "predict":
        _handle_flags(argv)
        return predict_model(argv)
    elif command == "download":
        return download_model(argv)

def wrap_command(command, **kw):
    argv = [sys.argv[0], command]
    argv += [f"--{key}={val}" for (key, val) in kw.items()]
    reset_flags()
    load_flags(command, argv=argv)
    return runner(argv)

def cli():
    def no_return_runner(argv):
        runner(argv)

    command = sys.argv[1]
    load_flags(command)
    app.run(no_return_runner)
