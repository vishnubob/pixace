import os
import sys
import gin
from absl import app

from . flags import FLAGS, load_flags
from . import tokens
from . import factory

def _bind_params(argv):
    gin.bind_parameter("pixace.tokenizer", FLAGS.tokenizer)
    gin.bind_parameter("pixace.model_name", FLAGS.model_name)
    gin.bind_parameter("pixace.weights_dir", FLAGS.weights_dir)
    gin.bind_parameter("pixace.model_type", FLAGS.model_type)

def _load_params(argv, config_filename="config.gin"):
    if FLAGS.model_name:
        config_dir = os.path.join(FLAGS.weights_dir, FLAGS.model_name)
    elif FLAGS.checkpoint:
        config_dir = os.path.split(os.path.abspath(FLAGS.checkpoint))[0]
    else:
        msg = f"You need to provide a model name or a checkpoint file"
        raise ValueError(msg)
    config_path = os.path.join(config_dir, config_filename)
    if not os.path.exists(config_path):
        msg = f"Can not find {config_filename} within {config_dir}"
        raise ValueError(msg)
    gin.parse_config_file(config_path)

def cli_runner(argv):
    command = argv[1]

    if command == "train":
        from . train import Trainer
        _bind_params(argv)
        try:
            _load_params(argv)
        except ValueError:
            pass
        Trainer._absl_main(argv)
    elif command == "predict":
        from . decode import Decoder
        _load_params(argv)
        Decoder._absl_main(argv)
    elif command == "download":
        from . zoo import ModelZoo
        ModelZoo._absl_main(argv)

def cli():
    command = sys.argv[1]
    load_flags(command)
    app.run(cli_runner)
