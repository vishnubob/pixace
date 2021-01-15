def download_model(model_name=None, checkpoint="default", weights_dir="model-weights"):
    from . zoo import ModelZoo
    zoo = ModelZoo(weights_dir=weights_dir)
    return zoo.download(model_name=model_name, checkpoint=checkpoint)

def get_trainer(
        model_name=None,
        model_type="reformer",
        weights_dir="model-weights",
        image_size=32,
        bitdepth=(5,4,4),
    ):

    from . train import Trainer
    return Trainer(
        model_name=model_name,
        model_type=model_type,
        weights_dir=weights_dir,
        image_size=image_size,
        bitdepth=bitdepth,
    )

def get_predictor(
        model_name=None,
        model_type="reformer",
        weights_dir="model-weights",
        checkpoint=None,
        image_size=32,
        bitdepth=(5,4,4)
    ):

    from . decode import Decoder
    return Decoder(
        model_name=model_name,
        model_type=model_type,
        weights_dir=weights_dir,
        image_size=image_size,
        bitdepth=bitdepth,
    )
