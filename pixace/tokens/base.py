import numpy as np

class Token(object):
    def __init__(self, name=None, value=None):
        assert name is not None and value is not None
        assert type(value) == int
        self.name = name
        self.value = value

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value

class TokenMap(object):
    def __init__(self, tokens=tuple()):
        self.tokens = tuple(tokens)
        self.names = {sp.name: sp.value for sp in self.tokens}
        self.values = {sp.value: sp.name for sp in self.tokens}
        assert len(self.names) == len(self.tokens)
        assert len(self.values) == len(self.tokens)

    @classmethod
    def default_tokens(cls, offset=0):
        defaults = ("pad", "bos", "eos", "unk")
        tokens = []
        for (value, name) in enumerate(defaults):
            tok = Token(name, value + offset)
            tokens.append(tok)
        return cls(tokens=tokens)
    
    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __contains__(self, key):
        return (key in self.names) \
            or (key in self.values)

    def __getitem__(self, key):
        if type(key) == int:
            return self.values[key]
        return self.names[key]

class TokenModel(object):
    def __init__(self, tokens=None, max_len=None, add_markers=True, offset=None, dtype=np.int32):
        assert max_len is not None
        if tokens is None:
            self.tokens = TokenMap.default_tokens()
        else:
            self.tokens = TokenMap(tokens)
        self.add_markers = add_markers
        self.max_len = max_len
        self.offset = offset if offset is not None else len(self.tokens)
        if self.add_markers:
            self.max_len += 2
        self.dtype = dtype

    @property
    def n_tokens(self):
        return len(self.tokens)

    def token_map(self):
        tmap = [tk.name for tk in self.tokens.tokens]
        return tuple(tmap)

    def add_bos_eos(self, ary):
        bos = int(self.tokens["bos"])
        eos = int(self.tokens["eos"])
        return np.pad(
            ary, 
            (1, 1), 
            constant_values=(bos, eos)
        )

    def filter_values(self, ary, values=None):
        values = values or \
            tuple(self.tokens.values.keys())
        for val in values:
            ary = ary[ary != val]
        return ary
    
    def pad(self, ary, max_len=None, side="right"):
        max_len = max_len or self.max_len
        if side == "right":
            pad_spec = (0, 1)
        elif side == "left":
            pad_spec = (1, 0)
        else:
            raise ValueError(side)
        pad_cnt = max_len - len(ary)
        if pad_cnt > 0:
            pad_shape = ary.shape[:-1] + (pad_cnt,)
            pad_value = int(self.tokens["pad"])
            pad = np.ones(pad_shape, dtype=ary.dtype) * pad_value
            ary = np.concatenate((ary, pad), axis=-1)
        return ary

    def trim(self, ary, max_len=None, side="right"):
        max_len = max_len or self.max_len
        if side == "right":
            return ary[:max_len]
        elif side == "left":
            return ary[max_len:]
        else:
            raise ValueError(side)

    def pad_or_trim(self, ary, max_len=None, trim_side="right", pad_side="right"):
        max_len = max_len or self.max_len
        if len(ary) < max_len:
            ary = self.pad(ary, max_len=max_len, side=pad_side)
        elif len(ary) > max_len:
            ary = self.trim(ary, max_len=max_len, side=trim_side)
        return ary
    
    def encode(self, ary):
        ary = ary + self.offset
        if self.add_markers:
            ary = self.add_bos_eos(ary)
        return ary.astype(self.dtype)

    def decode(self, ary):
        if self.add_markers:
            ary = self.filter_values(ary)
        ary = ary - self.offset
        return ary
