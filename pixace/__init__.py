def load_config(**kw):
    from . main import load_config
    load_config(**kw)

def get_factory(**kw):
    from . factory import get_factory
    return get_factory(**kw)

def download_model(model_name=None, checkpoint="default", weights_dir="model-weights"):
    from . zoo import ModelZoo
    zoo = ModelZoo(weights_dir=weights_dir)
    return zoo.download(model_name=model_name, checkpoint=checkpoint)

def get_trainer(**kw):
    from . train import Trainer
    return Trainer(**kw)

def get_decoder(**kw):
    from . decode import Decoder
    return Decoder(**kw)
