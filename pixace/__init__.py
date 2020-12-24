import sys
from . import tokens

def wrap_command(command, **kw):
    argv = [sys.argv[0], command]
    argv += [f"--{key}={val}" for (key, val) in kw.items()]
    sys.argv = argv

    # absl workaround
    old_exit = sys.exit
    def no_exit(val):
        return
    sys.exit = no_exit
    try:
        from . main import cli
        cli()
    finally:
        sys.exit = old_exit

def train(**kw):
    wrap_command("train", **kw)

def predict(**kw):
    wrap_command("predict", **kw)

def download(**kw):
    wrap_command("download", **kw)
