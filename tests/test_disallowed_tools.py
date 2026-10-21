#!/usr/bin/env python3

import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

from tests.helpers import ROOT

HARNESS_DIR = os.path.join(ROOT, "harness")
CODEX_AGENT = os.path.join(ROOT, "agents", "codex", "agent.py")
sys.path.insert(0, HARNESS_DIR)

from harness import (  # noqa: E402
    effective_disallowed_tools,
    find_disallowed_tool_use,
    web_access_track,
)

WEB = ["WebSearch", "WebFetch", "RemoteTrigger"]


def _jsonl(*rows):
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def _codex_search(action="open_page"):
    return {
        "type": "item.completed",
        "item": {"type": "web_search", "action": {"type": action}},
    }


def _claude_tool(name):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": name, "input": {}}]},
    }


class FindDisallowedToolUseTest(unittest.TestCase):
    def test_codex_web_search_flagged(self):
        trace = _jsonl(_codex_search(), _codex_search("search"), _codex_search())
        self.assertEqual(find_disallowed_tool_use(trace, WEB), {"web_search": 3})

    def test_started_items_not_double_counted(self):
        started = {"type": "item.started", "item": {"type": "web_search"}}
        trace = _jsonl(started, _codex_search())
        self.assertEqual(find_disallowed_tool_use(trace, WEB), {"web_search": 1})

    def test_claude_withheld_tool_flagged(self):
        trace = _jsonl(_claude_tool("WebFetch"), _claude_tool("Bash"))
        self.assertEqual(find_disallowed_tool_use(trace, WEB), {"WebFetch": 1})

    def test_clean_run(self):
        trace = _jsonl(_claude_tool("Bash"), {"type": "turn.started"})
        self.assertEqual(find_disallowed_tool_use(trace, WEB), {})

    def test_nothing_withheld_means_nothing_flagged(self):
        trace = _jsonl(_codex_search(), _claude_tool("WebFetch"))
        self.assertEqual(find_disallowed_tool_use(trace, []), {})

    def test_non_web_disallow_list_ignores_codex_search(self):
        self.assertEqual(
            find_disallowed_tool_use(_jsonl(_codex_search()), ["RemoteTrigger"]), {}
        )

    def test_tolerates_non_json_lines(self):
        trace = "{not json\nplain text\n" + _jsonl(_codex_search())
        self.assertEqual(find_disallowed_tool_use(trace, WEB), {"web_search": 1})


class WebTrackTest(unittest.TestCase):
    def test_closed_adds_web_tools_even_if_config_omits_them(self):
        got = effective_disallowed_tools(None, "closed")
        self.assertEqual(got, ["WebSearch", "WebFetch"])

    def test_closed_keeps_config_list_without_duplicates(self):
        got = effective_disallowed_tools(["WebSearch", "RemoteTrigger"], "closed")
        self.assertEqual(got, ["WebSearch", "RemoteTrigger", "WebFetch"])

    def test_open_drops_only_web_tools(self):
        got = effective_disallowed_tools(WEB + ["ListAgents"], "open")
        self.assertEqual(got, ["RemoteTrigger", "ListAgents"])

    def test_open_with_empty_config(self):
        self.assertEqual(effective_disallowed_tools(None, "open"), [])

    def test_open_book_run_is_not_tainted_by_web_use(self):
        withheld = effective_disallowed_tools(WEB, "open")
        self.assertEqual(
            find_disallowed_tool_use(_jsonl(_codex_search()), withheld), {}
        )

    def test_track_recorded(self):
        self.assertEqual(web_access_track("closed", False), "closed")
        self.assertEqual(web_access_track("open", False), "open")
        # an open container network defeats closed-book
        self.assertEqual(web_access_track("closed", True), "open")


class CodexWebSearchFlagTest(unittest.TestCase):
    def _flags(self, disallowed):
        # Load by path: every adapter is a module named "agent".
        spec = importlib.util.spec_from_file_location("codex_agent_under_test", CODEX_AGENT)
        codex_agent = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(os.path.dirname(CODEX_AGENT)))  # step_budget
        spec.loader.exec_module(codex_agent)

        a = codex_agent.Agent("c")
        a.disallowed_tools = disallowed
        captured = {}

        class _Stop(Exception):
            pass

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            raise _Stop

        with mock.patch.object(codex_agent.subprocess, "Popen", fake_popen), \
                mock.patch.object(a, "_exec"):
            try:
                a.run("prompt")
            except _Stop:
                pass
        return captured.get("cmd", [])

    def test_web_search_disabled_when_web_tools_withheld(self):
        cmd = self._flags(WEB)
        self.assertIn('web_search="disabled"', cmd)
        self.assertIn("--strict-config", cmd)

    def test_web_search_untouched_otherwise(self):
        cmd = self._flags([])
        self.assertNotIn('web_search="disabled"', cmd)
        self.assertNotIn("--strict-config", cmd)


if __name__ == "__main__":
    unittest.main()
