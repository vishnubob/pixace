import os
import piexif
import pprint
import pickle
import gzip
import trax
import jax
import numpy as np
from jax import numpy as jnp
from trax import layers as tl
from absl import logging
from PIL import Image
import gin
import uuid

from . utils import download_image_from_web
from . caption import caption_image

def get_uuid():
    return str(uuid.uuid4()).split('-')[1]

def autoreg(model, batch_size=1, inp=None, length=1, temperature=1.0, eos_id=None):
    if inp is not None:
        print(inp)

    stream = trax.supervised.decoding.autoregressive_sample_stream(
        model, 
        inputs=inp,
        batch_size=batch_size,
        temperature=temperature
    )

    result = []
    for sample in stream:
        sample = sample[:, None]
        result.append(sample)
        if len(result) > length:
            break
        if eos_id != None and np.all(np.any(result == eos_id, axis=-1)):
            break

    result = np.concatenate(result, axis=-1)

    if inp is not None:
        result = np.concatenate((inp, result), axis=-1)
    
    return result

def decode_batches(batches, tokenizer=None):
    for batch in batches:
        yield [tokenizer.decode(it) for it in batch] 

def export_gallery(batches, scale=None, tokenizer=None, outdir="gallery/public/images/photos"):
    os.makedirs(outdir, exist_ok=True)
    batches = decode_batches(batches, tokenizer)
    name = get_uuid()
    for (batch_idx, batch) in enumerate(batches):
        for (idx, item) in enumerate(batch):
            imgfn = f"{name}-b{batch_idx:02d}-{idx:02d}.jpg"
            imgfn = os.path.join(outdir, imgfn)
            caption = []
            images = []
            for val in item.values():
                if type(val) == Image.Image:
                    images.append(val)
                elif type(val) == str:
                    caption.append(val)
                else:
                    msg = f"Warning: unknown token type: {type(val)}"
                    print(msg)
            assert len(images) == 1
            img = images[0]
            caption = str.join('\n', caption)
            (width, height) = img.size
            size = (width * scale, height * scale)
            payload = dict()
            payload["0th"] = dict()
            payload["0th"][piexif.ImageIFD.XResolution] = (size[0], 1)
            payload["0th"][piexif.ImageIFD.YResolution] = (size[1], 1)
            payload["0th"][piexif.ImageIFD.ImageDescription] = caption
            payload = piexif.dump(payload)
            img = img.resize(size)
            img = caption_image(img, caption)
            img.save(imgfn, exif=payload)

# XXX: support more than just single image
def decode_output(img_list, batch_size=None, scale=None, tokenizer=None, image_key="image"):
    n_rows = len(img_list)
    assert n_rows > 0
    rows = []
    labels = []
    decodes = []
    for row in img_list:
        parts = [tokenizer.decode(it) for it in row] 
        decodes.append(parts)
        row = [np.array(it[image_key], dtype=np.uint8) for it in parts]
        row = [np.pad(it, ((1, 1), (1, 1), (0, 0))) for it in row]
        row = np.vstack(row)
        rows.append(row)
    tbl = np.hstack(rows)
    img = Image.fromarray(tbl)
    (width, height) = img.size
    size = (width * scale, height * scale)
    img = img.resize(size)
    return (img, decodes)

class Decoder(object):
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def max_len(self):
        return self.tokenizer.max_len
    
    @property
    def n_tokens(self):
        return self.tokenizer.n_tokens

    @classmethod
    def load(cls, tokenizer=None, **kw):
        model = load_model(**kw)
        return cls(model=model, tokenizer=tokenizer)

    def reset_model(self, batch_size=None):
        weights = self.model.weights
        example = np.zeros([batch_size, self.max_len]).astype(np.int32)
        signature = trax.shapes.signature(example)
        self.model.init(signature)
        self.model.weights = weights

    def predict(self, batch_size=1, prompts=None, cut=None, temperature=1, scale=256):
        output = []
        if prompts:
            batch_size = min(batch_size, len(prompts))
            if len(prompts) > batch_size:
                prompts = prompts[:batch_size]
            inp = [self.tokenizer.encode(pt, pad=False) for pt in prompts]
            mincut = min([len(enc) for enc in inp])
            cut = mincut if cut is None else min(cut, mincut)
            inp = np.vstack(inp)
            inp = inp[:, :cut]
            #pad = np.ones((batch_size, self.max_len - cut), dtype=np.int32) * self.tokenizer.tokens['pad']
            #actual = np.concatenate((inp, pad), axis=-1)
            #output.append(actual)
        else:
            inp = None
            cut = 0

        length = self.max_len - cut - 1

        for temp in temperature:
            temp = float(temp)
            self.reset_model(batch_size=batch_size)
            last_key = self.tokenizer.order[-1]
            eos_id = self.tokenizer.models[last_key].tokens["eos"]
            row = autoreg(self.model, batch_size=batch_size, inp=inp, length=length, temperature=temp, eos_id=eos_id)
            output.append(row)

        #return decode_output(output, batch_size=batch_size, scale=scale, tokenizer=self.tokenizer)
        export_gallery(output, scale=scale, tokenizer=self.tokenizer)

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS
        from . factory import get_factory

        factory = get_factory()
        model = factory.load_model(checkpoint=FLAGS.checkpoint, mode="predict")
        decoder = cls(model=model, tokenizer=factory.tokenizer)

        if isinstance(FLAGS.temperature, (int, float)):
            FLAGS.temperature = [FLAGS.temperature] 

        decoder.predict(
            batch_size=FLAGS.batch_size,
            temperature=FLAGS.temperature,
            prompts=FLAGS.prompt,
            cut=FLAGS.cut,
            scale=FLAGS.scale
        )
