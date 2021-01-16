import os
import numpy as np
import sentencepiece as spm

from . base import TokenModel

class TextTokenModel(TokenModel):
    def __init__(self, model_file="tokens.dat", **kw):
        super().__init__(offset=0, **kw)
        assert os.path.exists(model_file)
        self._model = spm.SentencePieceProcessor(model_file=model_file)

    @property
    def n_tokens(self):
        return super().n_tokens + self._model.vocab_size()

    @classmethod
    def build(cls, corpus=None, vocab_size=None, model_type="bpe", save_as="tokens.dat", force=False, **kw):
        assert not os.path.exists(save_as) or force

        idmap = {}
        ins = TokenModel(**kw)
        reserved = ("pad", "bos", "eos", "unk")
        for key in reserved:
            id_key = f"{key}_id"
            if key in ins.tokens:
                idmap[id_key] = ins.tokens[key]
            else:
                idmap[id_key] = None

        try:
            with open(save_as, 'wb') as fh:
                spm.SentencePieceTrainer.train(
                    sentence_iterator=iter(corpus),
                    model_type=model_type,
                    model_writer=fh, 
                    vocab_size=vocab_size,
                    **idmap
                )
        except:
            os.unlink(save_as)
            raise

        return cls(model_file=save_as, **kw)

    def encode(self, txt):
        encoded = self._model.encode(txt)
        encoded = self.trim(encoded, max_len=self.max_len - 2)
        encoded = np.array(encoded)
        encoded = super().encode(encoded)
        return encoded
    
    def decode(self, tokens):
        tokens = super().decode(tokens)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self._model.decode(tokens)
