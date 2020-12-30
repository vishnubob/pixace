import numpy as np
from . base import BaseTokenModel

class SerialTokenModel(BaseTokenModel):
    def __init__(self, models=None, order=None, max_len=None, **kw):
        super().__init__(**kw)
        self.models = models
        self.order = order
        self.max_len = max_len

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
            if key == "image":
                part = model.encode_image(part) + offset
            else:
                part = model.encode(part) + offset
            offset += model.n_tokens
            encoded.append(part)
        encoded = np.concatenate(encoded, axis=-1)
        padlen = self.max_len - len(encoded)
        if padlen > 0:
            pad = np.zeros((padlen, ), dtype=np.int32)
            encoded = np.concatenate((encoded, pad), axis=-1)
        return encoded

    def decode(self, toks):
        decoded = {}
        offset = 0
        for key in self.order:
            model = self.models[key]
            part = toks - offset
            part = part[part > 0]
            part = part[part < model.n_tokens]
            if key == "image":
                part = model.decode_image(part)
            else:
                part = model.decode(part)
            offset += model.n_tokens
            decoded[key] = part
        return decoded
