import os
import sys
import gin
from absl import app

from . flags import FLAGS, load_flags
from . import tokens

def _handle_flags(argv):
    from . import factory
    gin.bind_parameter("pixace.tokenizer", FLAGS.tokenizer)
    gin.bind_parameter("pixace.model_name", FLAGS.model_name)
    gin.bind_parameter("pixace.weights_dir", FLAGS.weights_dir)
    gin.bind_parameter("pixace.model_type", FLAGS.model_type)

def cli_runner(argv):
    command = argv[1]

    if command == "train":
        from . train import Trainer
        _handle_flags(argv)
        Trainer._absl_main(argv)
    elif command == "predict":
        from . decode import Decoder
        _handle_flags(argv)
        Decoder._absl_main(argv)
    elif command == "download":
        from . zoo import ModelZoo
        ModelZoo._absl_main(argv)

def cli():
    command = sys.argv[1]
    load_flags(command)
    app.run(cli_runner)
