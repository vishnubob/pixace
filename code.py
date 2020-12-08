import glob
from PIL import Image
import numpy as np
#import jax.numpy as np
#import jax
from sklearn.cluster import KMeans


class ColorCounts(object):
    def __init__(self, bits=8, channels=3):
        self.bits = bits
        self.n_colors = 2 ** self.bits
        self.channels = channels
        shape = tuple([self.n_colors for _ in range(self.channels)])
        self.counts = np.zeros(shape, dtype=np.uint32)
        self.counts = np.ravel(self.counts)
        print(self.counts.shape)

    def count(self, img):
        img = img.reshape((self.channels, -1)).astype(np.uint32)
        #next_total = img.shape[-1] + np.sum(self.counts)
        for ch in range(self.channels - 1):
            img[ch] <<= (8 * (self.channels - ch - 1))
        locs = np.sum(img, axis=0)
        self.counts[locs] += 1
        #assert(np.sum(self.counts) == next_total)

def to_blocks(ni, rows, cols):
    (w, h) = ni.shape[:2]
    assert w % cols == 0
    assert h % rows == 0
    blocks = [ni[col:col + cols, row:row + rows] for col in range(0, w, cols) for row in range(0, h, rows)]
    return blocks

def from_blocks(blocks, rows, cols):
    return np.block(blocks).reshape(cols, rows, -1)

def block_mean(blocks):
    ch = blocks[0].shape[-1]
    N = blocks[0].reshape(ch, -1).shape[1]
    blocks = [np.sum(bl.T.reshape(ch, -1), axis=-1) / N for bl in blocks]
    return blocks

def load_image(pt, sz=(256, 256)):
    img = Image.open(pt)
    img = img.resize(sz)
    img = img.convert('RGB')
    img = np.array(img)
    return img

def to_image(ni, sz=(256, 256)):
    o_img = np.round(ni * 255).astype(np.uint8)
    o_img = Image.fromarray(o_img)
    if sz:
        o_img = o_img.resize(sz)
    return o_img

def proc_img(pt, sz=(256, 256), rc=(16, 16)):
    (rows, cols) = rc
    ni = load_image(pt)
    b_ni = to_blocks(ni, rows, cols)
    b_ni = block_mean(b_ni)
    ni = from_blocks(b_ni, rows, cols)
    pixels = ni.reshape(rows * cols, -1)
    return pixels

def batch(pts, N=2 ** 11):
    stack = None
    for pt in pts:
        ary = proc_img(pt)
        if stack is None:
            stack = ary
        else:
            stack = np.vstack((stack, ary))
        if stack.shape[0] > N:
            yield stack
            stack = None

def to_colorspace(km, pt, new_fn, sz=(256, 256), rc=(16, 16)):
    ni = load_image(pt)
    (rows, cols) = rc
    b_ni = to_blocks(ni, rows, cols)
    b_ni = block_mean(b_ni)
    new_img = km.cluster_centers_[km.predict(b_ni)]
    new_img = from_blocks(new_img, rows, cols)
    new_img = to_image(new_img, sz=sz)
    new_img.save(new_fn)

m_pt = "/mnt/media/.i/monster.png"
pt = "/mnt/media/Datasets/flickr30k_images/flickr30k_images"
pts = glob.glob(pt + "/*.jpg")
#km = KMeans(n_clusters=2 ** 10, random_state=0)

cc = ColorCounts()
for (idx, pt) in enumerate(pts):
    img = load_image(pt)
    cc.count(img)
    print(idx)
    #km.fit(bt)
    #nm_pt = f"monster_{idx}.png"
    #to_colorspace(km, m_pt, nm_pt)
    #print(nm_pt)
