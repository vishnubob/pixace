import os
import numpy as np
import sentencepiece as spm

from . base import BaseTokenModel

class SentenceTokenModel(BaseTokenModel):
    def __init__(self, fn_model=None, dtype=np.int32, **kw):
        super().__init__(**kw)
        self._model = None
        self.dtype = dtype
        if fn_model is not None:
            self.load_model(fn_model)

    @property
    def n_tokens(self):
        return super().n_tokens + self._model.vocab_size()

    def load_model(self, fn_model=None):
        self._model = spm.SentencePieceProcessor(model_file=fn_model)

    def build(self, fn_model=None, corpus=None, vocab_size=None, model_type="bpe", force=False):
        assert not os.path.exists(fn_model) or force

        idmap = {}
        # defined by sentence piece
        reserved = ("pad", "bos", "eos", "unk")
        for key in reserved:
            id_key = f"{key}_id"
            if key in self.reserved:
                idmap[id_key] = self.reserved[key]
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

    def val_to_token(self, value):
        try:
            return self.val_to_token(value)
        except KeyError:
            pass
        return self.model.id_to_piece(value - self.offset)

    def token_to_val(self, token):
        try:
            return self.token_to_val(token)
        except KeyError:
            pass
        return self.model.piece_to_id(token) + self.offset

    def encode(self, txt):
        encoded = self._model.encode(txt)
        encoded = np.array(encoded, dtype=self.dtype)
        return encoded
    
    def decode(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self._model.decode(tokens)
