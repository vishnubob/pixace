import time
from absl import flags

FLAGS = flags.FLAGS

def common_flags():
    flags.DEFINE_string('model_name', None, help=('Model name.'))
    flags.DEFINE_string('model_type', "reformer", help=('Model type (transformer, reformer)'))
    flags.DEFINE_string('weights_dir', 'model-weights', help=('Top level directory for model data.'))
    flags.DEFINE_integer('batch_size', 16, help=('Batch size for training.'))
    # XXX: allow other aspect ratios
    #flags.DEFINE_integer('image_size', 32, help=('Edge size for square image.'))
    #flags.DEFINE_list('bitdepth', [5, 4, 4], help=('HSV bitdepths'))
    #flags.DEFINE_string('spm_model', None, help=('SPM model'))
    #flags.DEFINE_integer('max_len', None, help=('max len'))
    flags.DEFINE_multi_string('tokenizer', None, help=('Tokenizer spec, can be used more than once'))

    """
    @flags.validator("bitdepth", "bitdepth requires three comma seperated ints")
    def _validate_bitdepth(value):
        if len(value) != 3:
            return False
        digs = [
            (type(it) is int) or (type(it) is str and it.isdigit()) 
            for it in value
        ]
        return all(digs)
    """

def train_flags():
    common_flags()
    flags.DEFINE_integer('steps_per_epoch', 1000, help=('Number of steps per epochs'))
    flags.DEFINE_integer('steps_per_eval', None, help=('Number of steps per eval'))
    flags.DEFINE_integer('n_epochs', 100, help=('Number of epochs'))
    flags.DEFINE_string('train_data', "images", help=('JSON file containing a list of dicts'))
    flags.DEFINE_string('val_data', None, help=('JSON file containing a list of dicts'))

def predict_flags():
    common_flags()
    flags.DEFINE_string('checkpoint', None, help=('Path to checkpoint (default is latest)'))
    flags.DEFINE_list('prompt', [], help=('one or more prompt images, optional'))
    flags.DEFINE_integer('cut', None, help=('Cut input image at pixel #'))
    # XXX: normalize scale around 1 or 100
    flags.DEFINE_integer('scale', 256, help=('Scale output'))
    flags.DEFINE_string('out', "collage.png", help=('Where to save the prediction image'))
    flags.DEFINE_list('temperature', [1], help=('One or more temperature values used for predictions, can repeat the same value.'))

def download_flags():
    flags.DEFINE_string('model_name', None, help=('Name of the model'))
    flags.DEFINE_string('weights_dir', 'model-weights', help=('Top level directory for model data.'))
    flags.DEFINE_string('checkpoint', "default", help=('Name of the checkpoint'))

def load_flags(command, argv=None):
    if command == "train":
        train_flags()
    elif command == "predict":
        predict_flags()
    elif command == "download":
        download_flags()
    if argv is not None:
        flags.FLAGS(argv)
