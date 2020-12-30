import numpy as np

class BaseTokenModel(object):
    DefaultReservedTokens = {
        "pad": 0,
        "bos": 1,
        "eos": 2,
        "unk": 3,
    }

    def __init__(self, reserved=None, add_markers=True, offset=0):
        self.reserved = self.DefaultReservedTokens \
                if reserved is None else reserved
        self._add_markers = add_markers
        self._bos_id = self.reserved["bos"]
        self._eos_id = self.reserved["eos"]
        self.offset = offset

    def val_to_token(self, value):
        rev = {v: k for (k, v) in self.reserved}
        return rev[value]

    def token_to_val(self, token):
        return self.reserved[value]

    @property
    def n_tokens(self):
        return self.offset

    def encode(self, ary):
        ary = ary + self.offset
        # add bos and eos
        if self._add_markers:
            ary = np.pad(ary, (1, 1), 
                constant_values=(self._bos_id, self._eos_id))
        return ary

    def decode(self, toks):
        toks = toks - self.offset
        if self._add_markers:
            # filter out bos and eos
            toks = toks[toks != self._bos_id]
            toks = toks[toks != self._eos_id]
        return toks
