#!/usr/bin/env python3
"""Generic adapter for complete, third-party SRE-Bench agent images."""

import json
import os
import re
import subprocess
import sys

import yaml


RUNNER_PATH = "/usr/local/bin/srebench-agent"
PROTOCOL_VERSION = "1"
PROVIDER_SETTINGS = {
    "openai": {
        "hosts": ["api.openai.com"],
        "secrets": {"OPENAI_API_KEY": "OPENAI_API_KEY"},
        "environment": {"OPENAI_BASE_URL": "https://api.openai.com/v1"},
    },
    "anthropic": {
        "hosts": ["api.anthropic.com"],
        "secrets": {"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"},
        "environment": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
    },
    "google": {
        "hosts": ["generativelanguage.googleapis.com"],
        "secrets": {
            "GEMINI_API_KEY": "GEMINI_API_KEY",
            "GOOGLE_API_KEY": "GEMINI_API_KEY",
        },
        "environment": {
            "SREBENCH_GOOGLE_BASE_URL": "https://generativelanguage.googleapis.com"
        },
    },
}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _config_error(path, message):
    raise ValueError(f"invalid custom agent config {path}: {message}")


def load_custom_agent_config(path):
    """Load and validate the small, host-controlled custom-agent schema."""
    path = os.path.abspath(path)
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except OSError as exc:
        raise ValueError(f"cannot read custom agent config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse custom agent config {path}: {exc}") from exc

    if not isinstance(config, dict):
        _config_error(path, "top level must be a mapping")
    allowed_keys = {"version", "name", "image", "providers", "environment"}
    unknown = sorted(set(config) - allowed_keys)
    if unknown:
        _config_error(path, f"unknown field(s): {', '.join(unknown)}")
    if config.get("version") != 1:
        _config_error(path, "version must be 1")

    name = config.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        _config_error(
            path,
            "name must be 1-64 letters, digits, dots, underscores, or hyphens",
        )
    image = config.get("image")
    if (
        not isinstance(image, str)
        or not image.strip()
        or image.startswith("-")
        or "://" in image
        or any(c.isspace() for c in image)
    ):
        _config_error(path, "image must be a Docker/OCI image reference")

    providers = config.get("providers", [])
    if not isinstance(providers, list):
        _config_error(path, "providers must be a list")
    normalized_providers = []
    for provider in providers:
        if not isinstance(provider, str) or provider not in PROVIDER_SETTINGS:
            supported = ", ".join(PROVIDER_SETTINGS)
            _config_error(
                path,
                f"provider '{provider}' is not supported; choose from: {supported}",
            )
        if provider not in normalized_providers:
            normalized_providers.append(provider)

    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        _config_error(path, "environment must be a mapping")
    normalized_environment = {}
    managed_names = {"SREBENCH_PROVIDERS"}
    for settings in PROVIDER_SETTINGS.values():
        managed_names.update(settings["secrets"])
        managed_names.update(settings["environment"])
    for key, value in environment.items():
        if not isinstance(key, str) or not _ENV_RE.fullmatch(key):
            _config_error(path, f"environment key '{key}' is invalid")
        if key.startswith("SREBENCH_") or key in managed_names:
            _config_error(path, f"environment variable '{key}' is managed by SRE-Bench")
        if not isinstance(value, (str, int, float, bool)):
            _config_error(path, f"environment value for '{key}' must be a scalar")
        normalized_environment[key] = str(value)

    allowed_hosts = []
    normalized_secrets = {}
    normalized_environment["SREBENCH_PROVIDERS"] = ",".join(normalized_providers)
    for provider in normalized_providers:
        settings = PROVIDER_SETTINGS[provider]
        for host in settings["hosts"]:
            if host not in allowed_hosts:
                allowed_hosts.append(host)
        normalized_secrets.update(settings["secrets"])
        normalized_environment.update(settings["environment"])

    return {
        "version": 1,
        "name": name,
        "image": image.strip(),
        "providers": normalized_providers,
        "allowed_hosts": allowed_hosts,
        "secrets": normalized_secrets,
        "environment": normalized_environment,
        "config_path": path,
    }


def pull_custom_agent_image(image):
    """Pull an image and return its immutable ID for result reporting."""
    print(f"Pulling custom agent image {image}...")
    subprocess.run(["docker", "pull", image], check=True)
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    details = json.loads(result.stdout)[0]
    return details.get("Id")


class Agent:
    """Run the standard in-container SRE-Bench runner and parse its events."""

    def __init__(self, container_name, config):
        self.container_name = container_name
        self.config = config
        self.reasoning = ""
        self.last_message = ""
        self.model = None

    def _base_environment(self):
        env = dict(self.config["environment"])
        missing = []
        for target, source in self.config["secrets"].items():
            value = os.environ.get(source)
            if value is None:
                missing.append(source)
            else:
                env[target] = value
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise RuntimeError(
                f"missing custom agent secret environment variable(s): {names}"
            )
        return env

    def _exec(self, command, environment=None, **kwargs):
        docker_cmd = ["docker", "exec"]
        if "input" in kwargs:
            docker_cmd.append("-i")
        client_environment = os.environ.copy()
        for key, value in sorted((environment or {}).items()):
            # Let the Docker CLI read the value from its own environment. This
            # keeps API keys and other secrets out of the host process list.
            client_environment[key] = value
            docker_cmd += ["-e", key]
        docker_cmd.append(self.container_name)
        return subprocess.run(
            docker_cmd + command, env=client_environment, **kwargs
        )

    def auth(self):
        # Resolve secrets before starting the run, and fail with a useful error
        # before the agent executable receives the prompt.
        self._base_environment()
        result = self._exec(
            ["/bin/sh", "-c", f"test -x {RUNNER_PATH}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"custom image must provide executable {RUNNER_PATH}")

    def install_mcp(self, _tool_manifests):
        # Complete custom images own their tools and MCP configuration.
        return None

    def run(
        self,
        prompt,
        debug=False,
    ):
        environment = self._base_environment()
        environment["SREBENCH_PROTOCOL"] = PROTOCOL_VERSION

        env_flags = []
        client_environment = os.environ.copy()
        for key, value in sorted(environment.items()):
            client_environment[key] = value
            env_flags += ["-e", key]
        proc = subprocess.Popen(
            ["docker", "exec", "-i"] + env_flags + [self.container_name, RUNNER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=client_environment,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        output_lines = []
        llm_calls = 0
        for line in proc.stdout:
            output_lines.append(line)
            if debug:
                sys.stdout.write(line)
                sys.stdout.flush()
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "started" and isinstance(event.get("model"), str):
                self.model = event["model"]
            elif event_type == "llm_call":
                llm_calls += 1
                if not debug:
                    sys.stdout.write(f"\rLLM calls: {llm_calls}   ")
                    sys.stdout.flush()
            elif event_type == "final":
                text = event.get("text")
                if isinstance(text, str):
                    self.last_message = text.strip()

        proc.wait()
        if not debug:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.reasoning = "".join(output_lines)
        return self.reasoning, self.last_message, llm_calls, 0

    def get_metrics(self):
        return None

    def cleanup(self):
        return None
