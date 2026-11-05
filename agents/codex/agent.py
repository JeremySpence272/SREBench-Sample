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


CALL_TYPES = {"custom_tool_call", "function_call", "local_shell_call"}
CALL_OUTPUT_TYPES = {"custom_tool_call_output", "function_call_output", "local_shell_call_output"}


def trim_rollout_tail(lines, drop_steps=1):
    """Remove the last `drop_steps` tool steps of the last turn from a codex rollout.

    A step is the reasoning item(s) plus the tool call and its output. The records after
    the last turn boundary (turn_context) are scanned; the cut starts at the first
    response_item after the output that precedes the dropped steps, and the turn's
    task_complete event is kept so the turn still reads as closed. Returns
    (kept_lines, dropped_call_summaries); untouched when nothing can be dropped.
    """
    recs = []
    for line in lines:
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            recs.append(None)
    turn_start = max(
        (i for i, r in enumerate(recs) if r and r.get("type") == "turn_context"), default=0
    )
    calls = [
        i for i in range(turn_start, len(recs))
        if recs[i] and recs[i].get("type") == "response_item"
        and (recs[i].get("payload") or {}).get("type") in CALL_TYPES
    ]
    if drop_steps <= 0 or not calls:
        return lines, []
    first_dropped_call = calls[-min(drop_steps, len(calls))]
    prev_output = max(
        (i for i in range(turn_start, first_dropped_call)
         if recs[i] and recs[i].get("type") == "response_item"
         and (recs[i].get("payload") or {}).get("type") in CALL_OUTPUT_TYPES),
        default=None,
    )
    if prev_output is None:
        # the first step of the turn: cut after the turn's user message
        prev_output = max(
            (i for i in range(turn_start, first_dropped_call)
             if recs[i] and recs[i].get("type") == "response_item"
             and (recs[i].get("payload") or {}).get("role") == "user"),
            default=first_dropped_call - 1,
        )
    cut = next(
        (i for i in range(prev_output + 1, first_dropped_call + 1)
         if recs[i] and recs[i].get("type") == "response_item"),
        first_dropped_call,
    )
    dropped = []
    for i in calls:
        if i >= cut:
            p = recs[i].get("payload") or {}
            text = p.get("input") or p.get("arguments") or json.dumps(p.get("action", ""))
            dropped.append(str(text)[:200])
    kept = lines[:cut]
    closing = [
        i for i in range(cut, len(recs))
        if recs[i] and recs[i].get("type") == "event_msg"
        and (recs[i].get("payload") or {}).get("type") == "task_complete"
    ]
    if closing:
        rec = dict(recs[closing[-1]])
        if "ordinal" in rec:
            rec["ordinal"] = cut
        kept = kept + [json.dumps(rec) + "\n"]
    return kept, dropped

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

    FLAG_MARKER = "flagged for possible cybersecurity"
    MAX_FRUITLESS_RESUMES = 3
    FLAG_RESUME_PROMPT = os.environ.get("SREBENCH_FLAG_RESUME_PROMPT") or (
        "The provider's content filter incorrectly classified the current task and cut off your previous turn; the task itself is "
        "unchanged and everything you wrote under /workspace is still there. Continue "
        "from where you left off, keeping your work strictly within the task as stated, "
        "and write your submission before you finish."
    )

    def run(
        self,
        prompt,
        max_steps=None,
        max_reasoning=None,
        max_tool_timeout=None,
        debug=False,
        count_llm_calls=False,
        resume_on_flag=0,
        resume_drop_steps=1,
    ):
        """Run codex and return (reasoning, last_message, steps_used, reasoning_used).

        With resume_on_flag=N, a turn the provider ends with its cybersecurity content
        flag is resumed in the same thread (up to N times) with FLAG_RESUME_PROMPT; step
        and reasoning counts accumulate across the resumed turns and the budget applies.
        Before each resume the flagged turn's last resume_drop_steps tool steps are
        removed from the thread's rollout (0 keeps the thread intact).
        """
        env_flags = []
        for k, v in self.ENV.items():
            env_flags += ["-e", f"{k}={v}"]
        model_flags = []
        # Native web search happens on Codex's servers, outside the container
        # network. Disable it explicitly for closed-book runs.
        if set(self.disallowed_tools) & WEB_TOOLS:
            model_flags += ["--strict-config", "--config", 'web_search="disabled"']
        common_flags = [
            "--yolo",
            "--json",
            "--output-last-message",
            RESULT_PATH,
        ]
        st = {
            "steps": 0,
            "reasoning": 0,
            "llm_calls": 0,
            "awaiting_llm": True,
            "last_event": time.monotonic(),
            "timed_out": False,
            "thread_id": None,
            "flagged": False,
            "hit_limit": False,
        }
        output_lines: list[str] = []
        argv = ["codex", "exec"] + model_flags + common_flags + ["-"]
        turn_prompt = prompt
        resumes = 0
        fruitless = 0  # consecutive resumed turns that were flagged before any tool step
        while True:
            steps_before = st["steps"]
            self._run_turn(
                argv, turn_prompt, env_flags, st, output_lines,
                max_steps, max_reasoning, max_tool_timeout, debug, count_llm_calls,
            )
            fruitless = fruitless + 1 if st["steps"] == steps_before else 0
            if (
                st["hit_limit"]
                or not st["flagged"]
                or st["thread_id"] is None
                or resumes >= resume_on_flag
            ):
                break
            if fruitless >= self.MAX_FRUITLESS_RESUMES:
                print(
                    f"\n[harness] {fruitless} consecutive turns flagged before any tool step; "
                    "the prompt itself is refused, giving up"
                )
                output_lines.append(json.dumps({"type": "harness.resume_abandoned",
                                                "fruitless_turns": fruitless}) + "\n")
                break
            resumes += 1
            dropped = self._trim_flagged_tail(st["thread_id"], resume_drop_steps, resumes)
            note = {
                "type": "harness.resume",
                "attempt": resumes,
                "thread_id": st["thread_id"],
                "reason": "provider content filter ended the turn",
                "dropped_steps": dropped,
            }
            print(
                f"\n[harness] content filter ended the turn; dropped {len(dropped)} "
                f"step(s) and resuming thread {st['thread_id']} ({resumes}/{resume_on_flag})"
            )
            output_lines.append(json.dumps(note) + "\n")
            argv = (
                ["codex", "exec", "resume"] + model_flags + common_flags
                + [st["thread_id"], "-"]
            )
            turn_prompt = self.FLAG_RESUME_PROMPT
            if dropped:
                turn_prompt += (
                    f" Your most recent {len(dropped)} command(s) and their output were "
                    "removed from this conversation; if you still need that result, "
                    "approach it differently."
                )

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

        budget = st["llm_calls"] if count_llm_calls else st["steps"]
        return reasoning, last_message, budget, st["reasoning"]

    def _trim_flagged_tail(self, thread_id, drop_steps, attempt):
        """Drop the flagged turn's last steps from the thread's rollout in the container.

        The original file is kept on the host under sessions_dir/trim-backups. Returns
        the dropped calls' input summaries (empty when nothing was changed).
        """
        if drop_steps <= 0:
            return []
        found = self._exec(
            ["sh", "-c",
             f"ls /root/.codex/sessions/*/*/*/rollout-*-{thread_id}.jsonl 2>/dev/null | head -1"],
            capture_output=True, text=True,
        )
        remote = found.stdout.strip()
        if not remote:
            print("[harness] rollout file not found; resuming with the thread intact")
            return []
        backups = os.path.join(self.sessions_dir, "trim-backups")
        os.makedirs(backups, exist_ok=True)
        local = os.path.join(backups, f"{os.path.basename(remote)}.before-resume-{attempt}")
        cp = subprocess.run(
            ["docker", "cp", f"{self.container_name}:{remote}", local],
            capture_output=True, text=True,
        )
        if cp.returncode:
            print(f"[harness] could not copy rollout out: {cp.stderr.strip()}")
            return []
        with open(local) as f:
            lines = f.readlines()
        kept, dropped = trim_rollout_tail(lines, drop_steps)
        if not dropped:
            return []
        trimmed = local + ".trimmed"
        with open(trimmed, "w") as f:
            f.writelines(kept)
        cp = subprocess.run(
            ["docker", "cp", trimmed, f"{self.container_name}:{remote}"],
            capture_output=True, text=True,
        )
        if cp.returncode:
            print(f"[harness] could not write trimmed rollout: {cp.stderr.strip()}")
            return []
        return dropped

    def _run_turn(
        self, argv, turn_prompt, env_flags, st, output_lines,
        max_steps, max_reasoning, max_tool_timeout, debug, count_llm_calls,
    ):
        """One `codex exec` (or `codex exec resume`) invocation; updates st in place."""
        st["flagged"] = False
        st["awaiting_llm"] = True
        proc = subprocess.Popen(
            ["docker", "exec", "-i"] + env_flags + [self.container_name] + argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(turn_prompt)
        proc.stdin.close()
        stdout = proc.stdout

        def progress():
            if not debug:
                sys.stdout.write(
                    f"\rStep {st['steps']} | Reasoning {st['reasoning']}   "
                )
                sys.stdout.flush()

        def read_output():
            for line in stdout:
                st["last_event"] = time.monotonic()
                if debug:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                output_lines.append(line)
                event_type, item_type = self._parse_event(line)
                if event_type == "thread.started":
                    try:
                        st["thread_id"] = json.loads(line).get("thread_id") or st["thread_id"]
                    except json.JSONDecodeError:
                        pass
                if event_type in ("error", "turn.failed") and self.FLAG_MARKER in line:
                    st["flagged"] = True
                # one LLM invocation per model-output turn: a tool start or an
                # agent_message that begins a new turn (parallel tools collapse to one).
                model_output = (
                    event_type == "item.started" and item_type in self.STEP_TYPES
                ) or (event_type == "item.completed" and item_type == "agent_message")
                if model_output and st["awaiting_llm"]:
                    st["llm_calls"] += 1
                    st["awaiting_llm"] = False
                    if count_llm_calls and over_budget(
                        self.container_name, st["llm_calls"], max_steps
                    ):
                        return
                if event_type != "item.completed":
                    continue
                if item_type in self.STEP_TYPES:
                    st["awaiting_llm"] = True  # tool result consumed; next output is a new turn
                    st["steps"] += 1
                    if not count_llm_calls and over_budget(
                        self.container_name, st["steps"], max_steps
                    ):
                        return
                    progress()
                elif item_type == "agent_message":
                    st["reasoning"] += 1
                    progress()
                    if max_reasoning is not None and st["reasoning"] >= max_reasoning:
                        return

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        if max_tool_timeout is not None:
            while reader.is_alive():
                reader.join(timeout=5)
                if (
                    reader.is_alive()
                    and (time.monotonic() - st["last_event"]) > max_tool_timeout
                ):
                    st["timed_out"] = True
                    print(
                        f"\n[harness] Tool timeout: no output for {max_tool_timeout}s, killing agent."
                    )
                    break
        else:
            reader.join()

        if not debug:
            sys.stdout.write("\n")
            sys.stdout.flush()

        budget = st["llm_calls"] if count_llm_calls else st["steps"]
        st["hit_limit"] = (
            st["timed_out"]
            or (max_steps is not None and budget >= max_steps)
            or (max_reasoning is not None and st["reasoning"] >= max_reasoning)
        )
        if st["hit_limit"]:
            subprocess.run(
                ["docker", "kill", self.container_name],
                capture_output=True,
            )
            proc.kill()
        proc.wait()

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
