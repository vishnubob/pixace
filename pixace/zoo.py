import sys
import os
import requests

ModelDirectory = {
    "animalfaces": {
        "default": "1Uh7Cd-kjYLhs7SjlFROKuUyT9Vv8h-5i"
    },
}

# goog drive downloader adapted from:
#  https://github.com/saurabhshri/gdrive-downloader
#

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, bufsize=2**15):
    with open(destination, "wb") as f:
        for (n_chunk, chunk) in enumerate(response.iter_content(bufsize)):
            if not chunk: 
                continue
            f.write(chunk)
            if (n_chunk + 1) % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        print()

def is_gdrive_model(model_name=None):
    return model_name in ModelDirectory

def is_gdrive_checkpoint(model_name=None, checkpoint="default"):
    return is_gdrive_model(model_name) and (checkpoint in ModelDirectory[model_name])

def get_checkpoint(model_name=None, checkpoint="default", weights_dir=None):
    weights_dir = weights_dir or "model-weights"

    if not is_gdrive_model(model_name):
        msg = f"{model_name} is not in the model list {str(list(ModelDirectory.keys()))}"
        raise KeyError(msg)

    if not is_gdrive_checkpoint(model_name, checkpoint):
        msg = f"{checkpoint} is not in the checkpoint list {str(list(ModelDirectory[model_name].keys()))}"
        raise KeyError(msg)

    gid = ModelDirectory[model_name][checkpoint]

    root = os.path.join(weights_dir, model_name)
    os.makedirs(root, exist_ok=True)
    if checkpoint == "default":
        fn = f"model.pkl.gz"
    else:
        fn = f"{checkpoint}_model.pkl.gz"
    fn = os.path.join(root, fn)
    msg = f"Downloading '{model_name}' to '{fn}'"
    print(msg)
    download_file_from_google_drive(gid, fn)
    return fn

class ModelZoo(object):
    def __init__(self, weights_dir="model-weights"):
        self.weights_dir = weights_dir

    def download(self, model_name=None, checkpoint="default"):
        checkpoint_path = get_checkpoint(model_name, checkpoint, self.weights_dir)
        return checkpoint_path

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS
        zoo = cls(weights_dir=FLAGS.weights_dir)
        zoo.download(model_name=FLAGS.model_name, checkpoint=FLAGS.checkpoint)

