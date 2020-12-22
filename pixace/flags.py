import time
from absl import flags

_get_ts = lambda: time.strftime("%m%d_%H%M")
modes = ['train', 'predict']

if 'FLAGS' not in locals():
    flags.DEFINE_enum('mode', 'train', modes, help='mode (train or predict)')
    flags.DEFINE_string('model_dir', 'model-weights', help=('Top level directory for model data.'))
    flags.DEFINE_string('run_name', _get_ts(), help=('Run name.'))
    flags.DEFINE_integer('batch_size', 16, help=('Batch size for training.'))
    flags.DEFINE_integer('steps_per_epoch', 1000, help=('number of epochs'))
    flags.DEFINE_integer('n_epochs', 10000, help=('Total number of epochs'))
    flags.DEFINE_integer('image_size', 22, help=('Edge size for square image.'))
    flags.DEFINE_list('bitdepth', [5, 4, 4], help=('HSV bitdepths'))
    flags.DEFINE_string('images', "images", help=('Path to images used for training'))
    flags.DEFINE_string('checkpoint', None, help=('Path to checkpoint to load'))
    flags.DEFINE_list('temps', [0], help=('Temperature(s) for predictions.'))
    flags.DEFINE_list('predict_input', [], help=('List of images to prompt predictions with'))
    flags.DEFINE_integer('cut', None, help=('Cut input at'))
    flags.DEFINE_integer('scale', 256, help=('Scale output by'))
    flags.DEFINE_string('out', "collage.png", help=('Where to save the prediction image'))

@flags.validator("bitdepth", "bitdepth requires three comma seperated ints")
def _validate_bitdepth(value):
    if len(value) != 3:
        return False
    digs = [
        (type(it) is int) or (type(it) is str and it.isdigit()) 
        for it in value
    ]
    return all(digs)

FLAGS = flags.FLAGS
