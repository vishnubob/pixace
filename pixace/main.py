import sys
import os
from absl import app

from . flags import FLAGS, load_flags, reset_flags
from . train import Trainer
from . inference import Inference
from . zoo import ModelZoo

def _handle_flags(argv):
    FLAGS.bitdepth = [int(bit) for bit in FLAGS.bitdepth]

def cli_runner(argv):
    command = argv[1]

    if command == "train":
        _handle_flags(argv)
        Trainer._absl_main(argv)
    elif command == "predict":
        _handle_flags(argv)
        Inference._absl_main(argv)
    elif command == "download":
        ModelZoo._absl_main(argv)

def cli():
    command = sys.argv[1]
    load_flags(command)
    app.run(cli_runner)
