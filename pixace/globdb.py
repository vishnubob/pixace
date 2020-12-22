import os
from pathlib import Path
from fnmatch import fnmatch

class GlobDatabase(object):
    def __init__(self, dbpath=None, pattern=None):
        self.dbpath = dbpath
        self.pattern = pattern
        self._load_db()

    def _load_db(self):
        assert os.path.exists(self.dbpath)
        assert self.pattern is not None
        print(f"Loading files from '{self.dbpath}' matching '{self.pattern}'")

        self._db = {}
        match_img = lambda fn: fnmatch(fn, self.pattern)

        for (root, dirs, files) in os.walk(self.dbpath):
            root = Path(root)
            for fn in files:
                if not match_img(fn):
                    continue
                pt = root.joinpath(fn)
                assert pt.stem not in self._db
                self._db[pt.stem] = pt

    def select(self, group=None):
        db = self._db.copy()
        if group:
            pat = f"*{group}*"
            toc = [it for it in db.items() if fnmatch(it[1], pat)]
            toc = dict(toc)
        return toc
