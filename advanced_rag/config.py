"""
Configuration Loader
Parses YAML config and returns structured configs for pipeline and components.
"""

from typing import Any, Dict, Optional
from dataclasses import asdict
import os

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None

from .pipeline import PipelineConfig


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load configuration files. Please install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(path: str) -> PipelineConfig:
    """
    Load PipelineConfig from YAML file, falling back to defaults for missing keys.
    """
    config_dict = load_yaml_config(path)
    pipeline_cfg = config_dict.get("pipeline", {}) or {}
    return PipelineConfig(**pipeline_cfg)


def load_component_configs(path: str) -> Dict[str, Any]:
    """
    Load all component config sections from YAML for external wiring.
    Returns a dictionary with keys: milvus, chunking, embeddings, reranking, evaluation, domains, monitoring, storage, security
    """
    cfg = load_yaml_config(path)
    # Environment variable interpolation for "${VAR}" patterns is out-of-scope here
    return {
        "milvus": cfg.get("milvus", {}),
        "chunking": cfg.get("chunking", {}),
        "embeddings": cfg.get("embeddings", {}),
        "reranking": cfg.get("reranking", {}),
        "evaluation": cfg.get("evaluation", {}),
        "domains": cfg.get("domains", {}),
        "monitoring": cfg.get("monitoring", {}),
        "storage": cfg.get("storage", {}),
        "security": cfg.get("security", {}),
    }


