import os
import time
import trax
import gin

get_timestamp = lambda: time.strftime("%m%d_%H%M")

def adjust_reformer(scope="decode", chunk_len=None, n_hashes=4):
    assert chunk_len
    gin_path = f"{scope}/trax.layers.SelfAttention.chunk_len"
    gin.bind_parameter(gin_path, chunk_len)
    gin_path = f"{scope}/LSHSelfAttention.n_hashes"
    gin.bind_parameter(gin_path, n_hashes)

class PixaceFactory(object):
    DefaultWeightsDir = "model-weights"

    def __init__(self, model_name=None, weights_dir=None, model_type=None, n_tokens=None, max_len=None, tokenizer=None):
        self.model_name = model_name or get_timestamp()
        self.weights_dir = weights_dir or self.DefaultWeightsDir
        self.model_type = model_type
        self.tokenizer = tokenizer

    @property
    def output_dir(self):
        return os.path.join(self.weights_dir, self.model_name)

    @property
    def max_len(self):
        return self.tokenizer.max_len
    
    @property
    def n_tokens(self):
        return self.tokenizer.n_tokens

    def init_model(self, mode="train"):
        msg = f"Initializing {self.model_type} model (n_tokens={self.n_tokens}, max_len={self.max_len})"
        print(msg)
        if self.model_type == "transformer":
            model = trax.models.TransformerLM(self.n_tokens, max_len=self.max_len, mode=mode)
        elif self.model_type == "reformer":
            model = trax.models.ReformerLM(self.n_tokens, max_len=self.max_len, mode=mode)
        else:
            msg = f"Unknown model type '{self.model_type}'"
            raise ValueError(msg)
        return model

    def load_model(self, checkpoint=None, scope="decode", mode="predict"):
        if checkpoint is None:
            checkpoint = os.path.join(self.output_dir, "model.pkl.gz")
        if self.model_type == "reformer" and mode == "predict":
            adjust_reformer(scope=scope, chunk_len=self.max_len)
            with gin.config_scope(scope):
                model = self.init_model(mode=mode)
        else:
            model = self.init_model(mode=mode)
        msg = f"Loading {self.model_type} model from '{checkpoint}'"
        print(msg)
        model.init_from_file(checkpoint, weights_only=True)
        return model

@gin.configurable('pixace')
def get_factory(model_name=None, weights_dir=None, model_type=None, tokenizer=None):
    from . tokens.factory import parse_and_build_tokenizer
    tokenizer = parse_and_build_tokenizer(tokenizer)

    factory = PixaceFactory(
        model_name=model_name,
        weights_dir=weights_dir,
        model_type=model_type,
        tokenizer=tokenizer
    )
    return factory
