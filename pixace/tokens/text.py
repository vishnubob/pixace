import os
import numpy as np
import sentencepiece as spm

from . base import TokenModel

class TextTokenModel(TokenModel):
    def __init__(self, spm_model=None, **kw):
        super().__init__(**kw, offset=0)
        self._model = None
        if spm_model is not None:
            self.load_model(spm_model)

    @property
    def n_tokens(self):
        return super().n_tokens + self._model.vocab_size()

    def load_model(self, spm_model=None):
        self._model = spm.SentencePieceProcessor(model_file=spm_model)

    def build(self, fn_model=None, corpus=None, vocab_size=None, model_type="bpe", force=False):
        assert not os.path.exists(fn_model) or force

        idmap = {}
        # defined by sentence piece
        reserved = ("pad", "bos", "eos", "unk")
        for key in reserved:
            id_key = f"{key}_id"
            if key in self.reserved:
                idmap[id_key] = self.tokens[key]
            else:
                idmap[id_key] = None

        try:
            with open(fn_model, 'wb') as fh:
                spm.SentencePieceTrainer.train(
                    sentence_iterator=iter(corpus),
                    model_type=model_type,
                    model_writer=fh, 
                    vocab_size=vocab_size,
                    **idmap
                )
        except:
            os.unlink(fn_model)
            raise

        self.load_model(fn_model)

    def encode(self, txt):
        encoded = self._model.encode(txt)
        encoded = self.trim(encoded, max_len=self.max_len - 2)
        encoded = super().encode(encoded)
        return encoded
    
    def decode(self, tokens):
        tokens = super().decode(tokens)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self._model.decode(tokens)
