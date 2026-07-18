import os

from models.momentum_model import MomentumModel
from models.volume_model import VolumeModel


def test_momentum_model_saves_weights_after_learn(tmp_path):
    path = str(tmp_path / "momentum.pkl")
    model = MomentumModel(weights_file=path)
    assert model.n_examples == 0
    model.learn({"price_momentum_1m": 5.0, "price_momentum_5m": -3.0}, True)
    assert model.n_examples == 1
    assert os.path.exists(path)


def test_momentum_model_loads_weights_on_start(tmp_path):
    path = str(tmp_path / "momentum.pkl")
    model = MomentumModel(weights_file=path)
    for _ in range(5):
        model.learn({"price_momentum_1m": 1.0, "price_momentum_5m": 1.0}, True)
    assert model.n_examples == 5

    reloaded = MomentumModel(weights_file=path)
    assert reloaded.n_examples == 5


def test_volume_model_loads_and_saves(tmp_path):
    path = str(tmp_path / "volume.pkl")
    model = VolumeModel(weights_file=path)
    model.learn({"volume_ratio": 2.0, "bid_ask_imbalance": 0.1}, False)
    assert os.path.exists(path)

    reloaded = VolumeModel(weights_file=path)
    assert reloaded.n_examples == 1
