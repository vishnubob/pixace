import numpy as np
from . base import BaseTokenModel

class SerialTokenModel(BaseTokenModel):
    def __init__(self, models=None):
        self.models = models

    @property
    def n_tokens(self):
        n_tokens = sum([m.n_tokens for m in self.tokens])
        return super().n_tokens + n_tokens

    def encode(self, parts): 
        assert len(parts) == len(self.models)
        encoded = []
        offset = 0
        for (model, part) in zip(self.models, parts):
            part = model.encode(part) + offset
            offset += model.n_tokens
            encoded.append(part)
        encoded = np.concatenate(encoded, axis=-1)
        return encoded

    def decode(self, toks):
        decoded = []
        offset = 0
        for model in self.models:
            part = toks - offset
            part = part[part > 0]
            part = part[part < model.n_tokens]
            part = model.decode(part)
            offset += model.n_tokens
            decoded.append(part)
        return decoded
