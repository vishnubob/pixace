def train(**kw):
    from . main import wrap_command
    wrap_command("train", **kw)

def predict(**kw):
    from . main import wrap_command
    wrap_command("predict", **kw)

def download(**kw):
    from . main import wrap_command
    wrap_command("download", **kw)
