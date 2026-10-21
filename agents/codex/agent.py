import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import shutil
import time

CODEX_DIR = os.path.expanduser("~/.codex")
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AGENT_DIR))
from step_budget import over_budget  # noqa: E402

# paths inside the container
REASONING_PATH = "/workspace/reasoning.jsonl"
RESULT_PATH = "/workspace/result.json"

# Challenge-config tool names for server-side web access. Codex exposes both
# WebSearch and WebFetch through its single native web_search tool.
WEB_TOOLS = {"WebSearch", "WebFetch"}


class Agent:
    def __init__(self, container_name):
        self.container_name = container_name
        self.model = None
        self.sessions_dir = tempfile.mkdtemp(prefix="revbench-codex-sessions-")
        # Set by the harness when the selected web track or challenge config
        # withholds server-side tools.
        self.disallowed_tools = []

    ENV = {"CODEX_UNSAFE_ALLOW_NO_SANDBOX": "1"}

    # Hosts required for Codex CLI API calls, auth, and telemetry.
    ALLOWED_HOSTS = [
        "api.openai.com",
        "auth.openai.com",
        "chatgpt.com",
        "ab.chatgpt.com",
    ]

    def _exec(self, cmd, **kwargs):
        """Run a command inside the container."""
        env_flags = []
        for k, v in self.ENV.items():
            env_flags += ["-e", f"{k}={v}"]
        return subprocess.run(
            ["docker", "exec"] + env_flags + [self.container_name] + cmd,
            **kwargs,
        )

    def auth(self):
        """Copy host ~/.codex into the container for auth,
        then overlay a fresh sessions dir to isolate this run."""
        subprocess.run(
            ["docker", "cp", CODEX_DIR, f"{self.container_name}:/root/.codex"],
            check=True,
            capture_output=True,
        )
        self._exec(["rm", "-rf", "/root/.codex/sessions"], capture_output=True)
        self._exec(["mkdir", "-p", "/root/.codex/sessions"], capture_output=True)

    def install_mcp(self, tool_manifests):
        """Register MCP servers from tool manifests with codex."""
        for m in tool_manifests:
            if "mcp" not in m:
                continue
            mcp = m["mcp"]
            cmd = ["codex", "mcp", "add", m["name"], "--", mcp["command"]]
            cmd += mcp.get("args", [])
            self._exec(cmd, check=True, capture_output=True)

    STEP_TYPES = {"command_execution", "mcp_tool_call"}

    def _parse_event(self, line):
        """Parse a JSONL line and return (event_type, item_type) or (None, None)."""
        try:
            obj = json.loads(line)
            return obj.get("type"), obj.get("item", {}).get("type")
        except (json.JSONDecodeError, AttributeError):
            return None, None

    def run(
        self,
        prompt,
        max_steps=None,
        max_reasoning=None,
        max_tool_timeout=None,
        debug=False,
        count_llm_calls=False,
    ):
        """Run codex and return (reasoning, last_message, steps_used, reasoning_used)."""
        env_flags = []
        for k, v in self.ENV.items():
            env_flags += ["-e", f"{k}={v}"]
        model_flags = []
        # Native web search happens on Codex's servers, outside the container
        # network. Disable it explicitly for closed-book runs.
        if set(self.disallowed_tools) & WEB_TOOLS:
            model_flags += ["--strict-config", "--config", 'web_search="disabled"']
        proc = subprocess.Popen(
            ["docker", "exec", "-i"]
            + env_flags
            + [
                self.container_name,
                "codex",
                "exec",
            ]
            + model_flags
            + [
                "--yolo",
                "--json",
                "--output-last-message",
                RESULT_PATH,
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        stdout = proc.stdout
        output_lines: list[str] = []
        steps = 0
        reasoning_count = 0
        llm_calls = 0
        awaiting_llm = (
            True  # a model-output turn is pending (start, or after a tool result)
        )
        last_event_time = time.monotonic()
        timed_out = False

        def read_output():
            nonlocal steps, reasoning_count, llm_calls, awaiting_llm, last_event_time
            for line in stdout:
                last_event_time = time.monotonic()
                if debug:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                output_lines.append(line)
                event_type, item_type = self._parse_event(line)
                # one LLM invocation per model-output turn: a tool start or an
                # agent_message that begins a new turn (parallel tools collapse to one).
                model_output = (
                    event_type == "item.started" and item_type in self.STEP_TYPES
                ) or (event_type == "item.completed" and item_type == "agent_message")
                if model_output and awaiting_llm:
                    llm_calls += 1
                    awaiting_llm = False
                    if count_llm_calls and over_budget(
                        self.container_name, llm_calls, max_steps
                    ):
                        return
                if event_type != "item.completed":
                    continue
                if item_type in self.STEP_TYPES:
                    awaiting_llm = (
                        True  # tool result consumed; next output is a new turn
                    )
                    steps += 1
                    if not count_llm_calls and over_budget(
                        self.container_name, steps, max_steps
                    ):
                        return
                    if not debug:
                        sys.stdout.write(
                            f"\rStep {steps} | Reasoning {reasoning_count}   "
                        )
                        sys.stdout.flush()
                elif item_type == "agent_message":
                    reasoning_count += 1
                    if not debug:
                        sys.stdout.write(
                            f"\rStep {steps} | Reasoning {reasoning_count}   "
                        )
                        sys.stdout.flush()
                    if max_reasoning is not None and reasoning_count >= max_reasoning:
                        return

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        if max_tool_timeout is not None:
            while reader.is_alive():
                reader.join(timeout=5)
                if (
                    reader.is_alive()
                    and (time.monotonic() - last_event_time) > max_tool_timeout
                ):
                    timed_out = True
                    print(
                        f"\n[harness] Tool timeout: no output for {max_tool_timeout}s, killing agent."
                    )
                    break
        else:
            reader.join()

        if not debug:
            sys.stdout.write("\n")
            sys.stdout.flush()

        budget = llm_calls if count_llm_calls else steps
        hit_limit = (
            timed_out
            or (max_steps is not None and budget >= max_steps)
            or (max_reasoning is not None and reasoning_count >= max_reasoning)
        )
        if hit_limit:
            subprocess.run(
                ["docker", "kill", self.container_name],
                capture_output=True,
            )
            proc.kill()

        proc.wait()
        reasoning = "".join(output_lines)

        # save reasoning trace to container workspace
        self._exec(
            ["sh", "-c", f"cat > {REASONING_PATH}"],
            input=reasoning,
            text=True,
            capture_output=True,
        )

        # read the last message (plain text from --output-last-message)
        last_message = ""
        r = self._exec(["cat", RESULT_PATH], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            last_message = r.stdout.strip()

        return reasoning, last_message, budget, reasoning_count

    # this was for a codex specific bug that caused the refresh token to be refreshed inside the container
    # once container was killed the auth token in host became misaligned
    # this should rarely be necessary as the refresh token refreshes ~every 8 days
    # can be used as an optional function for the future for agent specific cleanup that might be required
    def cleanup(self):
        """Save refreshed auth token back to host before container is destroyed."""
        subprocess.run(
            [
                "docker",
                "cp",
                f"{self.container_name}:/root/.codex/auth.json",
                os.path.join(CODEX_DIR, "auth.json"),
            ],
            capture_output=True,
        )

    def get_metrics(self):
        """Copy session logs out of the container and parse token usage."""
        subprocess.run(
            [
                "docker",
                "cp",
                f"{self.container_name}:/root/.codex/sessions/.",
                self.sessions_dir,
            ],
            capture_output=True,
        )

        logs = glob.glob(
            os.path.join(self.sessions_dir, "**", "*.jsonl"), recursive=True
        )
        tokens = None
        if logs:
            with open(max(logs, key=os.path.getmtime)) as f:
                for line in f:
                    # The codex --json stream omits the model; the session log
                    # carries it in session_meta/turn_context. Capture it here so
                    # the harness can record the actual model.
                    if self.model is None and '"model"' in line:
                        mm = re.search(r'"model"\s*:\s*"([^"]+)"', line)
                        if mm:
                            self.model = mm.group(1)
                    try:
                        d = json.loads(line)
                        p = d.get("payload", {})
                        if p.get("type") == "token_count" and "total_token_usage" in (
                            p.get("info") or {}
                        ):
                            tokens = p["info"]["total_token_usage"]
                    except (json.JSONDecodeError, KeyError):
                        pass

        return tokens

    @staticmethod
    def parse_reasoning(raw_text):
        """Parse Codex JSONL reasoning trace into normalized events.

        Returns list of dicts with keys: type, text, timestamp, metadata.
        """
        events = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = obj.get("type")

            if etype == "thread.started":
                events.append(
                    {
                        "type": "system",
                        "text": f"Thread started: {obj.get('thread_id', '')}",
                        "timestamp": None,
                        "metadata": {"subtype": "thread.started"},
                    }
                )

            elif etype == "turn.started":
                events.append(
                    {
                        "type": "system",
                        "text": "Turn started",
                        "timestamp": None,
                        "metadata": {"subtype": "turn.started"},
                    }
                )

            elif etype == "turn.completed":
                usage = obj.get("usage", {})
                events.append(
                    {
                        "type": "system",
                        "text": "Turn completed",
                        "timestamp": None,
                        "metadata": {"subtype": "turn.completed", "usage": usage},
                    }
                )

            elif etype in ("item.completed", "item.started"):
                item = obj.get("item", {})
                itype = item.get("type")
                status = item.get("status", "")

                if itype == "agent_message" and etype == "item.completed":
                    events.append(
                        {
                            "type": "reasoning",
                            "text": item.get("text", ""),
                            "timestamp": None,
                            "metadata": {"item_id": item.get("id")},
                        }
                    )

                elif itype == "command_execution":
                    command = item.get("command", "")
                    if etype == "item.started":
                        events.append(
                            {
                                "type": "tool_call",
                                "text": command,
                                "timestamp": None,
                                "metadata": {
                                    "tool_name": "bash",
                                    "item_id": item.get("id"),
                                    "input": command,
                                },
                            }
                        )
                    elif etype == "item.completed":
                        events.append(
                            {
                                "type": "tool_result",
                                "text": item.get("aggregated_output", ""),
                                "timestamp": None,
                                "metadata": {
                                    "tool_name": "bash",
                                    "item_id": item.get("id"),
                                    "exit_code": item.get("exit_code"),
                                    "status": status,
                                },
                            }
                        )

                elif itype == "mcp_tool_call" and etype == "item.completed":
                    events.append(
                        {
                            "type": "tool_call",
                            "text": json.dumps(
                                item.get("arguments", {}), ensure_ascii=False
                            ),
                            "timestamp": None,
                            "metadata": {
                                "tool_name": f"{item.get('server', '')}/{item.get('tool', '')}",
                                "item_id": item.get("id"),
                                "status": status,
                            },
                        }
                    )

                elif itype == "file_change" and etype == "item.completed":
                    changes = item.get("changes", [])
                    desc = ", ".join(
                        f"{c.get('kind', '?')} {c.get('path', '?')}" for c in changes
                    )
                    events.append(
                        {
                            "type": "tool_result",
                            "text": desc,
                            "timestamp": None,
                            "metadata": {
                                "tool_name": "file_change",
                                "item_id": item.get("id"),
                                "changes": changes,
                            },
                        }
                    )

        return events

    def invoke_judge(self, prompt, **kwargs):

        fd, temp_path = tempfile.mkstemp(prefix="codex-judge-out-")
        os.close(fd)
        empty_dir = tempfile.mkdtemp(prefix="codex-judge-dir-")

        try:
            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--yolo",
                    "--ephemeral",
                    "-C",
                    empty_dir,
                    "--output-last-message",
                    temp_path,
                    "-",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=500,
            )

            if os.path.exists(temp_path):
                with open(temp_path, "r") as f:
                    last_message = f.read().strip()
                if last_message:
                    result.stdout = last_message

            return result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(empty_dir):
                shutil.rmtree(empty_dir)
