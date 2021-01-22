import gin
from . bench import *

@bench()
def test_image_tokenizer():
    im = tokens.ImageTokenModel(image_size=(32, 32), bitdepth=(1, 1, 1))
    img = Image.open("test.png")
    e_img = im.encode_image(img)
    assert len(e_img) == (32 * 32 + 2)
    d_img = im.decode_image(e_img)
    assert np.all(np.array(d_img) == np.array(img))

@bench(with_corpus=True)
def test_text_tokenizer(corpus):
    tm = tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000)
    text = corpus[0]
    enc_text = tm.encode(text)
    dec_text = tm.decode(enc_text)
    assert text == dec_text

@bench(with_corpus=True)
def test_serial_tokenizer(corpus):
    tm = tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000)
    im = tokens.ImageTokenModel(image_size=(32, 32), bitdepth=(1, 1, 1))
    sm = tokens.SerialTokenModel(
        models={"IMAGE": im, "TEXT": tm},
        order=["IMAGE", "TEXT"]
    )

    text = corpus[0]
    img = "test.png"
    img = Image.open(img)
    item = {"TEXT": text, "IMAGE": img}
    
    encoded_item = sm.encode(item)
    decoded_item = sm.decode(encoded_item)
    assert decoded_item["TEXT"] == item["TEXT"]
    assert np.all(np.array(decoded_item["IMAGE"]) == np.array(item["IMAGE"]))

@bench(with_corpus=True)
def test_tokenizer_config(corpus):
    tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000, save_as="test.spm")

    tokenizers = [
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3',
        'type=text,key=label,model_file=test.spm,max_len=500',
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3'
    ]

    tokenizer = tokens.factory.parse_and_build_tokenizer(tokenizers)

@bench(with_corpus=True)
def test_tokenizer_gin_config(corpus):
    from . pixace.tokens import factory
    tokens.TextTokenModel.build(corpus, vocab_size=500, max_len=1000, save_as="test.spm")

    tokenizers = [
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3',
        'type=text,key=label,model_file=test.spm,max_len=500',
        'type=image,key=image,bitdepth=544,image_size=32x32,colorspace=rgb,n_channels=3'
    ]

    gin.bind_parameter("pixace.tokenizer", tokenizers)
    tokenizer = tokens.factory.parse_and_build_tokenizer()
    assert tokenizer
    #print("!!\n", gin.config_str(), "!!\n")
