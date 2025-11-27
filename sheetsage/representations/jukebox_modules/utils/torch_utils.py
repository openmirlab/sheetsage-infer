import gc
import torch as t

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def empty_cache():
    gc.collect()
    t.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

