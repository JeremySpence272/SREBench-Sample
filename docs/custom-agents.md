# Custom agent images

SRE-Bench can run a complete agent environment supplied as a Docker/OCI image.
The image owns the agent harness and every analysis tool it needs. SRE-Bench
does not install tools into a custom image, so `--tier` does not apply.

Custom images run on a dedicated Docker network restricted to the official API
hosts selected in their configuration. This requires Docker and permission to
run `sudo iptables` against Docker's `DOCKER-USER` chain. Built-in agents use
the same restriction when `--no-internet` is set.

Run a custom agent with:

```bash
python3 harness/harness.py <challenge> \
  --custom-agent-config custom-agent.yaml
```

`--custom-agent-config` cannot be combined with `--agent` or `--tier`. The
custom harness owns its step limits, retries, and internal timeouts, so
SRE-Bench's `--max-steps`, `--max-reasoning`, `--max-tool-timeout`, and
`--count-llm-calls` options do not apply. `--rounds` still works. The optional
`--max-wall-time SECONDS` is a host-side safety limit for each round and is
unlimited by default.

## Configuration schema

A complete configuration looks like this:

```yaml
version: 1
name: example-agent
image: docker.io/example/srebench-agent@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

providers:
  - openai
  - anthropic

environment:
  MODEL_NAME: example-model
```

### `version`

Required. The only supported value is `1`.

### `name`

Required. A short name used in logs, container names, and result filenames. It
may contain letters, numbers, `.`, `_`, and `-`, and must be at most 64
characters.

### `image`

Required. A Docker/OCI image reference, not a Docker Hub webpage URL. For
example:

```yaml
image: docker.io/example/srebench-agent:1.0
```

SRE-Bench pulls the image before every invocation. Docker registry credentials
come from the normal Docker credential store, so use `docker login` for private
images. Pinning an image by digest is recommended for repeatable benchmark
runs. The pulled image ID is recorded in the result.

### `providers`

Optional; defaults to an empty list. A custom agent may select any combination
of the initially supported providers:

| Config value | Model family | Official API host | Required host variable |
|---|---|---|---|
| `openai` | GPT | `api.openai.com` | `OPENAI_API_KEY` |
| `anthropic` | Claude | `api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `google` | Gemini | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` |

SRE-Bench derives the network allowlist and runner environment from this list.
Custom configs cannot provide arbitrary endpoints or secret mappings. More
provider endpoints are coming in future versions.

An empty list means the container receives no model API credentials and has no
external API endpoint access. Custom images always use this provider-only
allowlist. The restriction applies only to the dedicated SRE-Bench Docker
network; it does not disable networking for the host or unrelated containers.

For each selected provider, SRE-Bench reads the API key from the host variable
shown above and injects the normal variable into the runner. For Google, both
`GEMINI_API_KEY` and `GOOGLE_API_KEY` are populated from the host's
`GEMINI_API_KEY` for SDK compatibility. API keys never belong in this YAML file.
Because this lightweight design does not use a gateway, the custom image can
read every selected provider key. Use dedicated benchmark credentials with
appropriate spending limits and rotate them when needed.

### `environment`

Optional. Non-secret environment variables passed to the runner. Values must
be strings, numbers, or booleans. Names beginning with `SREBENCH_`, API-key
variables, and provider base-URL variables are managed by SRE-Bench and cannot
be overridden.

## Image contract

The image is a complete Linux agent environment. It must contain:

```text
/bin/sh
sleep
/usr/local/bin/srebench-agent
```

`/usr/local/bin/srebench-agent` must be executable. SRE-Bench ignores the
image's normal Docker entrypoint, starts the image as an idle container, and
then invokes this program. The container runs as UID 0, matching the existing
built-in agent containers.

The image must already contain everything the runner needs, including agent
CLIs, debuggers, decompilers, MCP servers, language runtimes, and supporting
libraries. Package installation during the benchmark run should not be
expected because general Internet access is blocked.

## Working directories

SRE-Bench mounts a temporary workspace at `/workspace`:

```text
/workspace/challenge/       challenge files and supporting material
/workspace/submission/      final files that will be graded
/workspace/available_tools.yaml
```

For a custom image, `available_tools.yaml` is empty because the image owns its
tools. The runner should use `/workspace` as its working directory and must
place the required answer under `/workspace/submission`, following the paths in
the challenge prompt.

The workspace is recreated between rounds unless `--keep-workspace` is used.

## Runner input

The runner receives the full, assembled challenge prompt as UTF-8 text on
standard input. It should read until end-of-file. It contains the challenge's
`prompt.md` and the exact submission instructions.

SRE-Bench also supplies these environment variables:

| Variable | Meaning |
|---|---|
| `SREBENCH_PROTOCOL` | Contract version; currently `1` |
| `SREBENCH_PROVIDERS` | Comma-separated selected providers |
| `OPENAI_BASE_URL` | Official OpenAI base URL when `openai` is selected |
| `ANTHROPIC_BASE_URL` | Official Anthropic base URL when `anthropic` is selected |
| `SREBENCH_GOOGLE_BASE_URL` | Official Gemini base URL when `google` is selected |

The runner is responsible for configuring its own agent and for passing the
prompt to it.

The `--web` option applies to built-in agent adapters. A custom image controls
its own web tools, so its web access is reported as `agent-managed`.

## Runner output

Standard output is a UTF-8 JSON Lines stream: one JSON object per line. The
runner should flush each event promptly so progress is visible. Plain
diagnostic logs may be written to standard error; they are saved in the raw
runner trace.

The supported events are:

```json
{"type":"started","model":"example-model-v1"}
{"type":"llm_call"}
{"type":"final","text":"Submission completed."}
```

- `started` reports that the harness launched; `model` is optional.
- `llm_call` is an optional, repeatable progress event emitted when the harness
  starts a model call.
- `final` reports that the harness finished; `text` is optional.

SRE-Bench may count `llm_call` events for its progress display, but it does not
use that count to control the custom agent. The custom harness chooses its own
tools, loops, step limits, and internal timeouts. If the benchmark operator
sets `--max-wall-time`, SRE-Bench stops the container when that total
agent-execution time is reached; this value is not passed into the runner.

The files under `/workspace/submission`, rather than the `final` event, are the
authoritative input to grading.

## Minimal Dockerfile shape

```dockerfile
FROM ubuntu:24.04

# Install the agent harness and every analysis tool here.
COPY srebench-agent /usr/local/bin/srebench-agent
RUN chmod +x /usr/local/bin/srebench-agent

WORKDIR /workspace
```

The `srebench-agent` program may be written in any language. It only needs to
read the prompt from standard input, run the chosen agent, emit the small JSON
Lines stream above, and leave the requested artifacts in
`/workspace/submission`.
