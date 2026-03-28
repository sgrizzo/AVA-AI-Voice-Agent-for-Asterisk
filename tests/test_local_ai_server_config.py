from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_local_ai_config_module():
    local_ai_dir = Path(__file__).resolve().parents[1] / "local_ai_server"
    sys.path.insert(0, str(local_ai_dir))
    try:
        return importlib.import_module("config")
    finally:
        # Avoid leaking path changes into other tests.
        if sys.path and sys.path[0] == str(local_ai_dir):
            sys.path.pop(0)


def test_llm_context_defaults_to_2048_on_gpu(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_CONTEXT", raising=False)
    monkeypatch.setenv("GPU_AVAILABLE", "1")
    config_mod = _load_local_ai_config_module()
    cfg = config_mod.LocalAIConfig.from_env()
    assert cfg.llm_context == 2048


def test_llm_context_defaults_to_768_on_cpu(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_CONTEXT", raising=False)
    monkeypatch.setenv("GPU_AVAILABLE", "0")
    config_mod = _load_local_ai_config_module()
    cfg = config_mod.LocalAIConfig.from_env()
    assert cfg.llm_context == 768


def test_llm_context_respects_env_override(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_CONTEXT", "1536")
    monkeypatch.setenv("GPU_AVAILABLE", "0")
    config_mod = _load_local_ai_config_module()
    cfg = config_mod.LocalAIConfig.from_env()
    assert cfg.llm_context == 1536


def test_tool_gateway_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("LOCAL_TOOL_GATEWAY_ENABLED", raising=False)
    config_mod = _load_local_ai_config_module()
    cfg = config_mod.LocalAIConfig.from_env()
    assert cfg.tool_gateway_enabled is True


def test_tool_gateway_enabled_respects_env_false(monkeypatch):
    monkeypatch.setenv("LOCAL_TOOL_GATEWAY_ENABLED", "0")
    config_mod = _load_local_ai_config_module()
    cfg = config_mod.LocalAIConfig.from_env()
    assert cfg.tool_gateway_enabled is False
