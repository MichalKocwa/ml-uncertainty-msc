"""Persistence helpers — save / load models to disk.

Sklearn-style models go through joblib, PyTorch models go through state_dict.
The directory is set globally via `set_save_dir()` once per notebook session.
"""
import os
import joblib
import torch


_SAVE_DIR = "."


def set_save_dir(path):
    """Set the directory used by all subsequent save/load calls."""
    global _SAVE_DIR
    _SAVE_DIR = path
    os.makedirs(path, exist_ok=True)


def get_save_dir():
    return _SAVE_DIR


def _path(name, ext):
    return os.path.join(_SAVE_DIR, f"{name}.{ext}")


# --------------------------------------------------------------------- #
# Sklearn-style models (BLR, GP, ...)
# --------------------------------------------------------------------- #
def save_sklearn(obj, name):
    joblib.dump(obj, _path(name, "joblib"))


def load_sklearn(name):
    return joblib.load(_path(name, "joblib"))


# --------------------------------------------------------------------- #
# PyTorch models
# --------------------------------------------------------------------- #
def save_nn(model, name):
    torch.save(model.state_dict(), _path(name, "pt"))


def load_nn(model_ctor, name):
    """model_ctor: zero-arg callable returning a fresh model with matching architecture."""
    model = model_ctor()
    model.load_state_dict(torch.load(_path(name, "pt"), map_location="cpu"))
    model.eval()
    return model


def save_nn_ensemble(models, name):
    torch.save([m.state_dict() for m in models], _path(name, "pt"))


def load_nn_ensemble(model_ctor, name):
    states = torch.load(_path(name, "pt"), map_location="cpu")
    models = []
    for s in states:
        m = model_ctor()
        m.load_state_dict(s)
        m.eval()
        models.append(m)
    return models
