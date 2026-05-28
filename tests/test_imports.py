from src.models import TeleopWMConfig, build_teleopwm_model


def test_public_model_api_imports():
    assert TeleopWMConfig is not None
    assert build_teleopwm_model is not None
