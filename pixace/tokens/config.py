from . serial import SerialTokenModel
from . text import TextTokenModel
from . image import ImageTokenModel

def parse_attr(keyval):
    assert '=' in keyval
    (key, val) = keyval.split('=')
    if key == "image_size":
        if 'x' in val:
            val = val.split('x')
            val = tuple(map(int, val))
        else:
            val = int(val)
            val = (val, val)
        assert len(val) == 2
    elif key == "bitdepth":
        assert len(val) == 3
        val = tuple(map(int, val))
    elif key == "colorspace":
        assert val in ('hsv', 'rgb')
    elif key == "type":
        assert val in ('image', 'text')
    elif key == "max_len":
        val = int(val)
    elif key == "n_channels":
        n_channels = int(val)
    return (key, val)

def parse_tokenizer(val):
    vals = val.split(',')
    attrs = dict(map(parse_attr, vals))
    assert "type" in attrs
    t_type = attrs["type"]
    if t_type == "image":
        allowed = ("type", "colorspace", "bitdepth", "image_size", "n_channels", "key")
    elif t_type == "text":
        allowed = ("type", "max_len", "key", "model_file")
    else:
        raise ValueError(t_type)
    bad_keys = set(attrs.keys()) - set(allowed)
    assert not bad_keys, f"not allowed {bad_keys}"
    cfg = {key: attrs.get(key) for key in allowed}
    return cfg

def build_tokenizer(tok_list):
    order = []
    models = {}
    for spec in tok_list:
        t_type = spec.pop("type")
        t_key = spec.pop("key", t_type)
        if t_type == "image":
            tok_model = ImageTokenModel(**spec)
        elif t_type == "text":
            tok_model = TextTokenModel(**spec)
        models[t_key] = tok_model
        order.append(t_key)
    return SerialTokenModel(models=models, order=order)

def parse_and_build_tokenizer(tok_list):
    tok_list = [parse_tokenizer(val) for val in tok_list]
    return build_tokenizer(tok_list)

