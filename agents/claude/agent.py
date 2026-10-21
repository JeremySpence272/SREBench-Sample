import json
import os
import subprocess
import sys
import threading
import tempfile
import time

CLAUDE_DIR = os.path.expanduser("~/.claude")
CLAUDE_JSON = os.path.expanduser("~/.claude.json")
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AGENT_DIR))
from step_budget import over_budget  # noqa: E402

# paths inside the container
REASONING_PATH = "/workspace/reasoning.jsonl"
RESULT_PATH = "/workspace/result.json"
JUDGE_SUBMISSION_PATH = "/workspace/submission"


class Agent:
    def __init__(self, container_name):
        self.container_name = container_name
        self.disallowed_tools = []

    ENV = {"IS_SANDBOX": "1"}

    # Hosts required for Claude Code API calls, auth, and telemetry.
    # Covers both direct API and AWS Bedrock modes.
    ALLOWED_HOSTS = [
        "api.anthropic.com",
        "platform.claude.com",
        # AWS Bedrock (region-specific endpoints constructed at runtime)
        "bedrock-runtime.us-east-1.amazonaws.com",
        "bedrock-runtime.us-east-2.amazonaws.com",
        "bedrock-runtime.us-west-1.amazonaws.com",
        "bedrock-runtime.us-west-2.amazonaws.com",
        "sts.amazonaws.com",
    ]

    def _exec(self, cmd, **kwargs):
        """Run a command inside the container."""
        docker_cmd = ["docker", "exec"]
        if "input" in kwargs:
            docker_cmd.append("-i")
        for k, v in self.ENV.items():
            docker_cmd += ["-e", f"{k}={v}"]
        docker_cmd.append(self.container_name)
        return subprocess.run(docker_cmd + cmd, **kwargs)

    def auth(self):
        """Copy host ~/.claude into the container for auth,
        then overlay a fresh sessions dir to isolate this run."""
        subprocess.run(
            [
                "docker",
                "cp",
                CLAUDE_DIR,
                f"{self.container_name}:/root/.claude",
            ],
            check=True,
            capture_output=True,
        )
        if os.path.isfile(CLAUDE_JSON):
            subprocess.run(
                [
                    "docker",
                    "cp",
                    CLAUDE_JSON,
                    f"{self.container_name}:/root/.claude.json",
                ],
                check=True,
                capture_output=True,
            )
        self._exec(["rm", "-rf", "/root/.claude/sessions"], capture_output=True)
        self._exec(["mkdir", "-p", "/root/.claude/sessions"], capture_output=True)
        self.install_deadline_hook()

    def install_deadline_hook(self):
        """Claude-specific delivery of the shared step-budget deadline.

        The bash BASH_ENV hook never re-fires in claude's persistent shell, so
        surface /workspace/.harness_deadline into claude's context via a
        PostToolUse hook (fires after every tool, any type). Codex is untouched.
        """
        hook = (
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "try: json.load(sys.stdin)\n"
            "except Exception: pass\n"
            'f = "/workspace/.harness_deadline"\n'
            "if os.path.exists(f):\n"
            "    try: msg = open(f).read().strip()\n"
            "    except Exception: msg = ''\n"
            "    if msg:\n"
            '        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))\n'
        )
        self._exec(
            ["sh", "-c", "cat > /root/.claude/deadline_hook.py"],
            input=hook, text=True, capture_output=True,
        )
        merge = (
            "import json, os\n"
            'p = "/root/.claude/settings.json"\n'
            "s = json.load(open(p)) if os.path.exists(p) else {}\n"
            'if not isinstance(s, dict): s = {}\n'
            's.setdefault("hooks", {})["PostToolUse"] = ['
            '{"matcher": "", "hooks": [{"type": "command", '
            '"command": "python3 /root/.claude/deadline_hook.py"}]}]\n'
            'json.dump(s, open(p, "w"))\n'
        )
        self._exec(["python3", "-c", merge], capture_output=True)

    def install_mcp(self, tool_manifests):
        """Register MCP servers from tool manifests with claude."""
        for m in tool_manifests:
            if "mcp" not in m:
                continue
            mcp = m["mcp"]
            cmd = ["claude", "mcp", "add", m["name"], "--", mcp["command"]]
            cmd += mcp.get("args", [])
            self._exec(cmd, check=True, capture_output=True)

    STEP_TYPES = {"tool_use"}
    # Wider deadline window than the shared default: claude can spawn sub-agents,
    # which need a turn to unwind and hand back before the main agent submits.
    DEADLINE_WARN_AT = 25
    # After the deadline fires, sub-agent tool calls stop counting toward the
    # budget (only the parent's steps do); this bounds total tool calls so a
    # runaway sub-agent tree still terminates.
    POST_DEADLINE_STEP_CAP = 250

    def _parse_event(self, line):
        """Parse a stream-json line and return (event_type, content_types, is_subagent).

        Claude Code emits one JSON object per line with a top-level "type" field.
        For "assistant" events, content_types is the set of block types in
        message.content (e.g. {"tool_use"}, {"text"}, {"thinking"}).
        For all other events, content_types is None.
        """
        try:
            obj = json.loads(line)
            event_type = obj.get("type")
            content_types = None
            # sub-agent (Agent-tool) events carry parent_tool_use_id at top level
            is_subagent = bool(obj.get("parent_tool_use_id"))
            if event_type == "assistant":
                msg = obj.get("message", {})
                content_types = {block.get("type") for block in msg.get("content", [])}
            return event_type, content_types, is_subagent
        except (json.JSONDecodeError, AttributeError):
            return None, None, False

    def run(
        self,
        prompt,
        max_steps=None,
        max_reasoning=None,
        max_tool_timeout=None,
        debug=False,
        count_llm_calls=False,
    ):
        """Run claude and return (reasoning, last_message, steps_used, reasoning_used)."""
        env_flags = []
        for k, v in self.ENV.items():
            env_flags += ["-e", f"{k}={v}"]
        claude_cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--verbose",
        ]
        if self.disallowed_tools:
            claude_cmd += ["--disallowedTools", ",".join(self.disallowed_tools)]
        proc = subprocess.Popen(
            ["docker", "exec", "-i"]
            + env_flags
            + [self.container_name]
            + claude_cmd,
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
        total = 0
        reasoning_count = 0
        llm_calls = 0
        awaiting_llm = (
            True  # a new assistant turn is pending (start, or after tool results)
        )
        last_event_time = time.monotonic()
        timed_out = False

        def read_output():
            nonlocal steps, total, reasoning_count, llm_calls, awaiting_llm, last_event_time
            for line in stdout:
                last_event_time = time.monotonic()
                if debug:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                output_lines.append(line)
                event_type, content_types, is_subagent = self._parse_event(line)
                if event_type == "user":
                    awaiting_llm = (
                        True  # tool results returned; next assistant is a new call
                    )
                    continue
                if event_type != "assistant" or content_types is None:
                    continue
                # one LLM API call per assistant turn; parallel tool_use blocks stream
                # as separate assistant events, so collapse them via the turn boundary.
                if awaiting_llm:
                    awaiting_llm = False
                    if not (max_steps is not None
                            and llm_calls >= max_steps - self.DEADLINE_WARN_AT
                            and is_subagent):
                        llm_calls += 1
                    if count_llm_calls and over_budget(
                        self.container_name, llm_calls, max_steps,
                        warn_at=self.DEADLINE_WARN_AT, subagent_aware=True,
                    ):
                        return
                if content_types & self.STEP_TYPES:
                    total += 1
                    # Once the deadline has fired, stop counting sub-agent tool
                    # calls so only the parent's steps deplete the budget -- this
                    # guarantees the parent its full warn_at window to submit.
                    deadline_fired = (max_steps is not None
                                      and steps >= max_steps - self.DEADLINE_WARN_AT)
                    if not (deadline_fired and is_subagent):
                        steps += 1
                    if (max_steps is not None
                            and total >= max_steps + self.POST_DEADLINE_STEP_CAP):
                        return
                    if not count_llm_calls and over_budget(
                        self.container_name, steps, max_steps,
                        warn_at=self.DEADLINE_WARN_AT, subagent_aware=True,
                    ):
                        return
                    if not debug:
                        sys.stdout.write(
                            f"\rStep {steps} | Reasoning {reasoning_count}   "
                        )
                        sys.stdout.flush()
                elif "text" in content_types:
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

        # extract last message from the result event and save to container
        # (claude has no --output-last-message flag, so we extract from the stream)
        last_message = ""
        for line in reversed(output_lines):
            try:
                obj = json.loads(line)
                if obj.get("type") == "result":
                    last_message = obj.get("result", "").strip()
                    break
            except (json.JSONDecodeError, AttributeError):
                pass

        if last_message:
            self._exec(
                ["sh", "-c", f"cat > {RESULT_PATH}"],
                input=last_message,
                text=True,
                capture_output=True,
            )

        # read the last message back from the container (mirrors codex pattern)
        r = self._exec(["cat", RESULT_PATH], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            last_message = r.stdout.strip()

        return reasoning, last_message, budget, reasoning_count

    # claude auth credentials may be refreshed inside the container;
    # copy them back to keep the host in sync.
    # can be used as an optional function for the future for agent specific cleanup that might be required
    def cleanup(self):
        """Save refreshed auth credentials back to host before container is destroyed."""
        subprocess.run(
            [
                "docker",
                "cp",
                f"{self.container_name}:/root/.claude/.credentials.json",
                os.path.join(CLAUDE_DIR, ".credentials.json"),
            ],
            capture_output=True,
        )

    def get_metrics(self):
        """Read reasoning trace from container and parse token usage."""
        r = self._exec(["cat", REASONING_PATH], capture_output=True, text=True)
        if r.returncode != 0:
            return None

        # look for the final "result" event which has cumulative usage
        tokens = None
        for line in r.stdout.splitlines():
            try:
                obj = json.loads(line)
                if obj.get("type") != "result":
                    continue
                usage = obj.get("usage", {})
                input_tokens = (
                    usage.get("input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                )
                tokens = {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": usage.get("cache_read_input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": input_tokens + usage.get("output_tokens", 0),
                }
            except (json.JSONDecodeError, KeyError):
                pass

        return tokens

    @staticmethod
    def parse_reasoning(raw_text):
        """Parse Claude stream-json reasoning trace into normalized events.

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
            ts = (
                obj.get("message", {}).get("usage", {})
                if etype == "assistant"
                else None
            )

            if etype == "system":
                subtype = obj.get("subtype", "")
                if subtype == "init":
                    model = obj.get("model", "unknown")
                    events.append(
                        {
                            "type": "system",
                            "text": f"Session started — model: {model}",
                            "timestamp": None,
                            "metadata": {"subtype": "init", "model": model},
                        }
                    )
                elif subtype in ("task_started", "task_progress", "task_notification"):
                    desc = obj.get("description", obj.get("prompt", ""))
                    events.append(
                        {
                            "type": "system",
                            "text": f"[{subtype}] {desc}",
                            "timestamp": None,
                            "metadata": {
                                "subtype": subtype,
                                "task_id": obj.get("task_id"),
                            },
                        }
                    )

            elif etype == "assistant":
                msg = obj.get("message", {})
                for block in msg.get("content", []):
                    btype = block.get("type")
                    if btype == "thinking":
                        events.append(
                            {
                                "type": "thinking",
                                "text": block.get("thinking", ""),
                                "timestamp": None,
                                "metadata": {},
                            }
                        )
                    elif btype == "text":
                        events.append(
                            {
                                "type": "reasoning",
                                "text": block.get("text", ""),
                                "timestamp": None,
                                "metadata": {},
                            }
                        )
                    elif btype == "tool_use":
                        inp = block.get("input", {})
                        if isinstance(inp, dict):
                            inp_str = json.dumps(inp, ensure_ascii=False)
                        else:
                            inp_str = str(inp)
                        events.append(
                            {
                                "type": "tool_call",
                                "text": inp_str,
                                "timestamp": None,
                                "metadata": {
                                    "tool_name": block.get("name", "unknown"),
                                    "tool_use_id": block.get("id"),
                                },
                            }
                        )

            elif etype == "user":
                msg = obj.get("message", {})
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    parts.append(item.get("text", str(item)))
                                else:
                                    parts.append(str(item))
                            content = "\n".join(parts)
                        events.append(
                            {
                                "type": "tool_result",
                                "text": str(content),
                                "timestamp": obj.get("timestamp"),
                                "metadata": {
                                    "tool_use_id": block.get("tool_use_id"),
                                    "is_error": block.get("is_error", False),
                                },
                            }
                        )

            elif etype == "result":
                result_text = obj.get("result", "")
                usage = obj.get("usage", {})
                events.append(
                    {
                        "type": "system",
                        "text": f"Session ended — {obj.get('subtype', '')}",
                        "timestamp": None,
                        "metadata": {
                            "subtype": "result",
                            "result_text": result_text,
                            "usage": usage,
                            "num_turns": obj.get("num_turns"),
                            "cost_usd": obj.get("total_cost_usd"),
                        },
                    }
                )

        return events

    def invoke_judge(self, prompt, challenge_dir, submission_dir):
        # submission files are already at /workspace/submission via the volume mount;
        # rewrite the host path in the prompt to the container path
        if submission_dir:
            prompt = prompt.replace(submission_dir, JUDGE_SUBMISSION_PATH)

        result = self._exec(
            [
                "claude",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                "-p",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=500,
        )

        return result
