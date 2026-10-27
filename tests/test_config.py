from sap_btp_buddy.config import configured_generation_model, live_generation_allowed
from sap_btp_buddy.generation_adapter import GenerationConfig


def test_project_config_enables_live_generation_and_matches_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n  allow_live_generation: true\n  generation_model: test/model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BTP_GENERATION_MODEL", "test/model")
    monkeypatch.setenv("BTP_GENERATION_API_KEY", "fixture-secret")
    assert live_generation_allowed(config_path) is True
    assert configured_generation_model(config_path) == "test/model"


def test_generation_config_loads_dotenv_values_without_network_call(monkeypatch):
    monkeypatch.setenv("BTP_GENERATION_API_KEY", "fixture-secret")
    monkeypatch.setenv("BTP_GENERATION_MODEL", "MiniMaxAI/MiniMax-M3.1-Flash")
    config = GenerationConfig.from_env()
    assert config.model_id == "MiniMaxAI/MiniMax-M3.1-Flash"
    assert config.base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert config.max_retries <= 2
