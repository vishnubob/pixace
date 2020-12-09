import os
import random
from pathlib import Path
from PIL import Image
from fnmatch import fnmatch

DatabasePath = "/mnt/media/Datasets/animal-faces"

class Animals(object):
    DefaultDatabasePath = DatabasePath

    def __init__(self, dbpath=None):
        self.dbpath = dbpath or self.DefaultDatabasePath
        self._load_db()

    def _load_db(self):
        self._db = {}
        pat = "*.jpg"
        match_img = lambda fn: fnmatch(fn, pat)

        for (root, dirs, files) in os.walk(self.dbpath):
            root = Path(root)
            for fn in files:
                if not match_img(fn):
                    continue
                pt = root.joinpath(fn)
                assert pt.stem not in self._db
                self._db[pt.stem] = pt

    def load_image(self, img_name):
        img = self._db[img_name]
        return Image.open(img)

    def get_cursor(self, group=None, shuffled=True):
        _db = self._db.copy()
        if group:
            pat = f"*{group}*"
            toc = [key for (key, val) in self._db.items() if fnmatch(val, pat)]
        else:
            toc = list(_db)
        if shuffled:
            random.shuffle(toc)
        for item in toc:
            yield lambda: self.load_image(item)

if __name__ == "__main__":
    a = Animals()
    itr = a.get_cursor(group='/train/')
    print(len([x.close() for x in itr]))

