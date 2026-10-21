#!/usr/bin/env python3

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from tests.helpers import ROOT

HARNESS_DIR = os.path.join(ROOT, "harness")
sys.path.insert(0, HARNESS_DIR)

from custom_agent import Agent, load_custom_agent_config  # noqa: E402


class TestCustomAgentConfig(unittest.TestCase):
    def _load(self, config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.safe_dump(config, f)
            f.flush()
            return load_custom_agent_config(f.name)

    def test_valid_config(self):
        config = self._load(
            {
                "version": 1,
                "name": "example-agent",
                "image": "docker.io/example/srebench-agent:1.0",
                "providers": ["openai", "anthropic", "openai"],
                "environment": {"MODEL_NAME": "example-v1", "DEBUG": False},
            }
        )
        self.assertEqual(config["name"], "example-agent")
        self.assertEqual(
            config["allowed_hosts"], ["api.openai.com", "api.anthropic.com"]
        )
        self.assertEqual(config["providers"], ["openai", "anthropic"])
        self.assertEqual(
            config["secrets"],
            {
                "OPENAI_API_KEY": "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            },
        )
        self.assertEqual(config["environment"]["DEBUG"], "False")
        self.assertEqual(
            config["environment"]["SREBENCH_PROVIDERS"], "openai,anthropic"
        )
        self.assertEqual(
            config["environment"]["OPENAI_BASE_URL"],
            "https://api.openai.com/v1",
        )

    def test_empty_provider_list_is_valid_and_means_no_external_access(self):
        config = self._load(
            {
                "version": 1,
                "name": "local-agent",
                "image": "example/local-agent:latest",
                "providers": [],
            }
        )
        self.assertEqual(config["allowed_hosts"], [])
        self.assertEqual(config["secrets"], {})

    def test_google_sets_both_common_key_names(self):
        config = self._load(
            {
                "version": 1,
                "name": "gemini-agent",
                "image": "example/agent:latest",
                "providers": ["google"],
            }
        )
        self.assertEqual(
            config["secrets"],
            {
                "GEMINI_API_KEY": "GEMINI_API_KEY",
                "GOOGLE_API_KEY": "GEMINI_API_KEY",
            },
        )
        self.assertEqual(
            config["allowed_hosts"], ["generativelanguage.googleapis.com"]
        )

    def test_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "provider 'other' is not supported"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "example/agent:latest",
                    "providers": ["other"],
                }
            )

    def test_rejects_non_list_providers(self):
        with self.assertRaisesRegex(ValueError, "providers must be a list"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "example/agent:latest",
                    "providers": "openai",
                }
            )

    def test_rejects_docker_hub_webpage_as_image(self):
        with self.assertRaisesRegex(ValueError, "Docker/OCI image reference"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "https://hub.docker.com/r/example/agent",
                }
            )

    def test_rejects_custom_endpoint_or_secret_mapping(self):
        for field, value in (
            ("network", {"allowed_endpoints": ["example.com"]}),
            ("secrets", {"API_KEY": "env:API_KEY"}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "unknown field"):
                    self._load(
                        {
                            "version": 1,
                            "name": "bad-agent",
                            "image": "example/agent:latest",
                            field: value,
                        }
                    )

    def test_rejects_provider_managed_environment(self):
        with self.assertRaisesRegex(ValueError, "managed by SRE-Bench"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "example/agent:latest",
                    "providers": ["openai"],
                    "environment": {"OPENAI_BASE_URL": "https://example.com"},
                }
            )

    def test_rejects_reserved_environment(self):
        with self.assertRaisesRegex(ValueError, "managed by SRE-Bench"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "example/agent:latest",
                    "environment": {"SREBENCH_MAX_STEPS": "999"},
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "unknown field"):
            self._load(
                {
                    "version": 1,
                    "name": "bad-agent",
                    "image": "example/agent:latest",
                    "volumes": ["/:/host"],
                }
            )


class _FakeInput:
    def __init__(self):
        self.value = ""

    def write(self, value):
        self.value += value

    def close(self):
        return None


class _FakeProcess:
    def __init__(self, lines):
        self.stdin = _FakeInput()
        self.stdout = iter(lines)
        self.returncode = None

    def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class TestCustomAgentProtocol(unittest.TestCase):
    def test_missing_secret_stops_before_run(self):
        config = {
            "environment": {},
            "secrets": {"MODEL_API_KEY": "SREBENCH_TEST_MISSING_SECRET"},
        }
        agent = Agent("test-container", config)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SREBENCH_TEST_MISSING_SECRET"):
                agent._base_environment()

    def test_reads_standard_event_stream(self):
        process = _FakeProcess(
            [
                '{"type":"started","model":"model-v1"}\n',
                '{"type":"llm_call"}\n',
                '{"type":"final","text":"done"}\n',
            ]
        )
        config = {
            "environment": {},
            "secrets": {},
        }
        agent = Agent("test-container", config)
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch(
                "custom_agent.subprocess.Popen", return_value=process
            ) as popen:
                reasoning, final, llm_calls, reasoning_count = agent.run("full prompt")

        self.assertEqual(process.stdin.value, "full prompt")
        self.assertIn('"type":"llm_call"', reasoning)
        self.assertEqual(final, "done")
        self.assertEqual(llm_calls, 1)
        self.assertEqual(reasoning_count, 0)
        self.assertEqual(agent.model, "model-v1")
        self.assertIsNone(agent.get_metrics())
        runner_environment = popen.call_args.kwargs["env"]
        self.assertEqual(runner_environment["SREBENCH_PROTOCOL"], "1")
        self.assertFalse(
            any(name.startswith("SREBENCH_MAX_") for name in runner_environment)
        )

    def test_secret_value_is_not_put_in_docker_command(self):
        process = _FakeProcess(['{"type":"final","text":"done"}\n'])
        config = {
            "environment": {},
            "secrets": {"MODEL_API_KEY": "HOST_MODEL_API_KEY"},
        }
        agent = Agent("test-container", config)
        with mock.patch.dict(
            os.environ, {"HOST_MODEL_API_KEY": "very-secret-value"}, clear=True
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch(
                    "custom_agent.subprocess.Popen", return_value=process
                ) as popen:
                    agent.run("prompt")

        command = popen.call_args.args[0]
        self.assertIn("MODEL_API_KEY", command)
        self.assertNotIn("very-secret-value", command)
        self.assertEqual(
            popen.call_args.kwargs["env"]["MODEL_API_KEY"], "very-secret-value"
        )


if __name__ == "__main__":
    unittest.main()
