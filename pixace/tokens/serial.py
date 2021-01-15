import numpy as np
from . base import TokenModel
from . image import ImageTokenModel
from . text import TextTokenModel

class SerialTokenModel(TokenModel):
    def __init__(self, models=None, order=None, **kw):
        self.models = models
        self.order = order
        max_len = sum([self.models[key].max_len for key in self.models])
        super().__init__(max_len=max_len, **kw)

    @property
    def n_tokens(self):
        n_tokens = sum([self.models[key].n_tokens for key in self.order])
        return super().n_tokens + n_tokens

    def encode(self, parts): 
        assert len(parts) == len(self.models)
        encoded = []
        offset = 0
        for key in self.order:
            (model, part) = (self.models[key], parts[key])
            if isinstance(model, ImageTokenModel):
                part = model.encode_image(part) + offset
            else:
                part = model.encode(part) + offset
            offset += model.n_tokens
            encoded.append(part)
        encoded = np.concatenate(encoded, axis=-1)
        encoded = self.pad(encoded)
        return encoded

    def decode(self, toks):
        decoded = {}
        offset = 0
        for key in self.order:
            model = self.models[key]
            part = toks - offset
            part = part[part > 0]
            part = part[part < model.n_tokens]
            if isinstance(model, ImageTokenModel):
                part = model.decode_image(part)
            else:
                part = model.decode(part)
            offset += model.n_tokens
            decoded[key] = part
        return decoded
