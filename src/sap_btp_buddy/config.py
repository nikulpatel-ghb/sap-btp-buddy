"""Configuration boundaries; secrets come from the environment, never notebooks."""

from dataclasses import dataclass
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    """Runtime paths and bounded retrieval settings."""

    index_dir: Path = Path("artifacts/index")
    top_k: int = 5
    model_name: str | None = None
    allow_live_generation: bool = False


def load_project_config(config_path: Path | None = None) -> dict[str, object]:
    """Load YAML settings and the project-local environment safely."""

    load_dotenv(PROJECT_ROOT / ".env")
    path = config_path or PROJECT_ROOT / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("config.yaml runtime section must be a mapping")
    return {"runtime": runtime, "raw": data}


def live_generation_allowed(config_path: Path | None = None) -> bool:
    """Return the explicit YAML gate for billable/provider-backed generation."""

    runtime = load_project_config(config_path)["runtime"]
    return bool(runtime.get("allow_live_generation", False))


def configured_generation_model(config_path: Path | None = None) -> str:
    """Return the environment model, checking YAML when it declares one."""

    runtime = load_project_config(config_path)["runtime"]
    model = os.getenv("BTP_GENERATION_MODEL", "").strip()
    declared = str(runtime.get("generation_model", "")).strip()
    if declared and declared not in {"PENDING", "NOT MEASURED"} and declared != model:
        raise ValueError("config.yaml generation_model does not match BTP_GENERATION_MODEL")
    return model


def generation_secret_configured() -> bool:
    """Return only whether the provider secret exists; never return its value."""

    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("BTP_GENERATION_API_KEY", "").strip())
