from . bench import *
from pixace.task import batch_generator

@bench()
def test_stop():
    im = tokens.ImageTokenModel(image_size=(32, 32), bitdepth=(1, 1, 1))
    tokenizer = tokens.SerialTokenModel(
        models={"image": im},
        order=["image"]
    )

    gen = pixace.task.batch_generator(
        "test.json",
        group="test",
        tokenizer=tokenizer,
        batch_size=32
    )

    itr = iter(gen)
    it = next(itr)
    gen.stop()
    assert len(it) == 3
    assert it[0].shape == (32, 32 * 32 + 2)
