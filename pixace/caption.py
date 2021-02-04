import os
import piexif
from PIL import Image, ImageDraw, ImageFont

class ImageTextWrapper(object):
    class Overflow(Exception):
        pass

    def textwrap(self, text, ff_overflow):
        (is_hovf, is_vovf) = ff_overflow
        wrapped = []
        line = []
        text = text.split(' ')
        for word in text:
            head = str.join('\n', wrapped)
            test_str = head + '\n' + str.join(' ', line + [word])
            if is_vovf(test_str):
                raise self.Overflow
            if is_hovf(test_str):
                wrapped.append(str.join(' ', line))
                line = [word]
            else:
                line.append(word)
        if line:
            wrapped.append(str.join(' ', line))
        text = str.join('\n', wrapped)
        if is_vovf(text):
            raise self.Overflow
        return text

    def size_and_wrap(self, text, bbox, font_filename, font_size=32):
        load_font = lambda sz: ImageFont.truetype(font_filename, size=sz)
        mat = Image.new("RGB", (bbox[2], bbox[3]))
        img_d = ImageDraw.Draw(mat)
        while font_size > 2:
            font = load_font(font_size)
            is_hovf = lambda tx: img_d.multiline_textbbox((0,0), tx, font=font)[-2] > mat.size[0]
            is_vovf = lambda tx: img_d.multiline_textbbox((0,0), tx, font=font)[-1] > mat.size[1]
            ff_ovf = (is_hovf, is_vovf)
            try:
                result = self.textwrap(text, ff_ovf)
                return (result, font)
            except self.Overflow:
                font_size -= 1
                continue
        raise self.Overflow

class ImageCaption(object):
    def __init__(self, height=None, margin=(2, 2, 2, 2), colors=('white', 'black'), font_filename='default.ttf', font_size=32):
        self.margin = margin
        self.font_filename = font_filename
        self.font_size = font_size
        self.colors = colors
        self.height = height 
        
    def get_margins(self, img, height):
        margins = (
            # left
            self.margin[0], 
            # top
            img.size[1] + self.margin[1], 
            # right
            img.size[0] - self.margin[2], 
            # bottom
            img.size[1] + height - self.margin[3]
        )
        return margins

    def get_textbox(self, img, height):
        bbox = (
            0, 0, 
            img.size[0] - (self.margin[0] + self.margin[2]),
            height - (self.margin[1] + self.margin[3])
        )
        return bbox

    def add_caption(self, img=None, text=None):
        height = int(round(img.size[1] * .15))  if self.height is None else self.height
        margins = self.get_margins(img, height)
        wrap = ImageTextWrapper()
        bbox = self.get_textbox(img, height)
        (text, font) = wrap.size_and_wrap(text, bbox, self.font_filename, self.font_size)
        mat = Image.new("RGB", (img.size[0], img.size[1] + height), self.colors[1])
        img_d = ImageDraw.Draw(mat)
        img_d.multiline_text((margins[0], margins[1]), text, self.colors[0], font=font)
        mat.paste(img, (0, 0))
        return mat

def load_image(fn):
    img = Image.open(fn)
    info = piexif.load(img.info["exif"])
    txt = info["0th"][270].decode()
    return (img, txt)

def adjust_syntax(text):
    text = text.split(' ')
    first = text[0].capitalize()
    newtxt = [first]
    for word in text[1:]:
        if word == ',':
            newtxt[-1] += ','
            continue
        if word[0] == '.':
            newtxt[-1] += '.'
            break
        if word[-1] == '.':
            break
        if word[0] == '\'':
            newtxt[-1] += word
            continue
        newtxt.append(word)
    return str.join(' ', newtxt)

def caption_image(img, caption, **kw):
    caption = adjust_syntax(caption)
    icap = ImageCaption(**kw)
    return icap.add_caption(img, caption)
