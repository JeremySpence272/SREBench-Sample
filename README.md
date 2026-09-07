# SREBench

A single-domain reverse-engineering benchmark: a coding agent is given a stripped-down, encoder-only file compressor and must reverse the format and recover the original inputs, scored 0–6.

## Requirements

- Python 3
- Docker (running)
- `pip install -r requirements.txt`  (pyyaml, pytest)
- An agent configured on the host: a working `~/.codex/`, `~/.claude/`, or `~/.gemini/`.

## Usage

List the available challenges:

```bash
python3 harness/harness.py --list-binaries
```

Run a challenge:

```bash
python3 harness/harness.py <challenge_key> --agent <agent> --tier <tier>
```

Example:

```bash
python3 harness/harness.py revdeflate/c-opt-sym-static --agent codex --tier tier2
```

## Arguments

| Arg | Description |
|---|---|
| `<challenge_key>` | Challenge to run, e.g. `revdeflate/c-opt-sym-static` (see `--list-binaries`) |
| `--agent` | Agent to run: `codex`, `claude`, or `gemini` |
| `--tier` | Tool tier: `tier1` (bash + runtimes), `tier2` (config's tools), `tier3` (all tools) |
| `--rounds N` | Number of independent runs (default 1) |
| `--early-stop` | Stop after the first successful round |
| `--max-steps N` | Max agent steps (default 200) |
| `--max-reasoning N` | Max agent reasoning messages (default unlimited) |
| `--max-tool-timeout N` | Max seconds for a single tool call before the agent is killed |
| `--count-llm-calls` | Count LLM API calls toward `--max-steps` instead of tool calls |
| `--no-internet` | Block all container networking except the agent's required API endpoints |
| `--keep-workspace` | Keep the run's workspace directory and print its path |
| `--debug` | Stream raw agent output instead of the refreshing status line |
| `--list-binaries` | List challenges and exit |

## Output

- `results/<id>.json` — run report (correctness, score, timing, tokens)
- `results/<id>_reasoning.jsonl` — the agent's reasoning trace
