from layer3_conscience.regime_detector import RegimeDetector, TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY, RANGE


# 12. Regime detector correctly identifies HIGH_VOLATILITY.
def test_high_volatility_detected_after_spike():
    detector = RegimeDetector(window=50)
    for _ in range(10):
        detector.detect(spot_momentum_15m=1.0, volatility_5m=2.0)  # calm baseline
    regime = detector.detect(spot_momentum_15m=1.0, volatility_5m=10.0)  # 5x the baseline
    assert regime == HIGH_VOLATILITY


def test_trending_up_when_momentum_positive_and_vol_low():
    detector = RegimeDetector(window=50)
    for _ in range(10):
        detector.detect(spot_momentum_15m=0.0, volatility_5m=5.0)
    regime = detector.detect(spot_momentum_15m=15.0, volatility_5m=4.0)
    assert regime == TRENDING_UP


def test_trending_down_when_momentum_negative_and_vol_low():
    detector = RegimeDetector(window=50)
    for _ in range(10):
        detector.detect(spot_momentum_15m=0.0, volatility_5m=5.0)
    regime = detector.detect(spot_momentum_15m=-15.0, volatility_5m=4.0)
    assert regime == TRENDING_DOWN


def test_range_when_no_data():
    detector = RegimeDetector(window=50)
    assert detector.detect(spot_momentum_15m=None, volatility_5m=None) == RANGE


def test_range_when_flat():
    detector = RegimeDetector(window=50)
    for _ in range(10):
        detector.detect(spot_momentum_15m=1.0, volatility_5m=5.0)
    regime = detector.detect(spot_momentum_15m=1.0, volatility_5m=5.0)
    assert regime == RANGE
