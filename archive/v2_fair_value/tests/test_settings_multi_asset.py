from config.settings import ASSETS, BINANCE_SYMBOLS, momentum_weights_path, volume_weights_path


def test_sol_included_in_assets_and_binance_symbols():
    assert "SOL" in ASSETS
    assert BINANCE_SYMBOLS.get("SOL") == "solusdt"


def test_per_asset_weight_paths_are_distinct():
    momentum_paths = {a: momentum_weights_path(a) for a in ASSETS}
    volume_paths = {a: volume_weights_path(a) for a in ASSETS}
    assert len(set(momentum_paths.values())) == len(ASSETS)
    assert len(set(volume_paths.values())) == len(ASSETS)
    assert momentum_paths["SOL"] != momentum_paths["BTC"] != momentum_paths["ETH"]
