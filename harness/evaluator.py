#!/usr/bin/env python3

import hashlib
import json
import os
import secrets
import stat
import subprocess
import tempfile


def evaluate(config, challenge_dir, submission_dir, reasoning_file, agent):
    """Dispatch evaluation based on config['evaluation'].

    return: {
        "correct": bool,
        "additional_info": dict,
    }
    """
    # `eval/test.sh` is the universal grader entrypoint. When `evaluation` is
    # omitted (or "artifact_test"/"binary"), the run is graded by its EXIT CODE
    # (0 == correct) — the default, covering any test you can script. A named
    # value opts into a grader that interprets the run differently (e.g. parses a
    # result file for partial scoring); see revdeflate_score.
    eval_type = config.get("evaluation") or "artifact_test"
    if eval_type in ("artifact_test", "binary"):
        return artifact_test(config, challenge_dir, submission_dir)
    elif eval_type == "revdeflate_score":
        return revdeflate_score(config, challenge_dir, submission_dir)
    else:
        raise ValueError(f"unknown evaluation type: {eval_type}")


# ── Generic grader: run eval/test.sh, exit code == pass/fail ──

EVAL_DOCKERFILE = """\
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip \\
    gcc libc6-dev \\
    socat openssl iptables sudo xxd jq ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /challenge
"""

_eval_image_tag = None


def _get_eval_image():
    """Return the generic eval image tag, building it if needed."""
    global _eval_image_tag
    if _eval_image_tag is not None:
        return _eval_image_tag
    digest = hashlib.sha256(EVAL_DOCKERFILE.encode()).hexdigest()[:12]
    tag = f"srebench-eval-{digest}"
    if subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True
    ).returncode != 0:
        print(f"Building eval image {tag}...")
        subprocess.run(
            ["docker", "build", "-t", tag, "-"],
            input=EVAL_DOCKERFILE.encode(),
            check=True,
        )
    _eval_image_tag = tag
    return tag


def _privilege_args(config):
    """Docker privilege flags from the per-challenge `eval` block.

    Defaults to `--privileged`; opt out with `eval.privileged: false` plus
    optional `eval.cap_add` / `eval.security_opt`.
    """
    ev = (config or {}).get("eval") or {}
    if ev.get("privileged", True):
        return ["--privileged"]
    args = []
    for cap in ev.get("cap_add", []) or []:
        args += ["--cap-add", cap]
    for opt in ev.get("security_opt", []) or []:
        args += ["--security-opt", opt]
    return args


def artifact_test(config, challenge_dir, submission_dir):
    """Generic pass/fail grader — the default.

    Runs the challenge's `eval/test.sh` in an isolated container with the
    challenge tree at /challenge and the agent's submission at /submission (also
    env SUBMISSION_DIR). Pass/fail is the script's EXIT CODE: 0 == correct.

    For graded/partial scoring, write your own grader (see revdeflate_score) that
    parses a result file, and add a dispatch case in evaluate() above.
    """
    script = os.path.join(challenge_dir, "eval", "test.sh")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"test script not found: {script}")
    if not os.access(script, os.X_OK):
        os.chmod(script, os.stat(script).st_mode | stat.S_IXUSR)

    eval_image = _get_eval_image()
    challenge_dir = os.path.abspath(challenge_dir)
    submission_dir = os.path.abspath(submission_dir)
    timeout = int((config or {}).get("eval", {}).get("timeout", 3600))

    container_name = f"srebench-eval-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        subprocess.run(
            [
                "docker", "create", "--name", container_name,
                *_privilege_args(config),
                "-e", "SUBMISSION_DIR=/submission",
                eval_image, "/challenge/eval/test.sh",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "cp", f"{challenge_dir}/.", f"{container_name}:/challenge"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["docker", "cp", f"{submission_dir}/.", f"{container_name}:/submission"],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["docker", "start", "-a", container_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    try:
        artifacts = sorted(os.listdir(submission_dir))
    except OSError:
        artifacts = []

    return {
        "correct": result.returncode == 0,
        "additional_info": {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip()[-2000:] if result.stdout else "",
            "stderr": result.stderr.strip()[-2000:] if result.stderr else "",
            "artifacts": artifacts,
        },
    }


_revdeflate_grader_tag = None


def _get_revdeflate_grader_image():
    """Build (or retrieve cached) the RevDeflate grader image.

    One challenge-agnostic image bakes the private ground truth (originals +
    manifest.json) plus the stdlib byte-comparison grader (grade.py/score.py). The
    per-variant encoder the agent reverses is NOT needed here: grading only
    byte-compares the recovered /submission tree against the private originals, and
    the challenge set is identical across every variant.
    """
    global _revdeflate_grader_tag
    if _revdeflate_grader_tag is not None:
        return _revdeflate_grader_tag

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ctx = os.path.join(root, "binaries", "revdeflate", "_grader")

    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(ctx):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            fp = os.path.join(dirpath, fn)
            h.update(os.path.relpath(fp, ctx).encode())
            with open(fp, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
    tag = f"revbench-revdeflate-grader-{h.hexdigest()[:12]}"

    result = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    if result.returncode == 0:
        print(f"RevDeflate grader image {tag} already exists, skipping build.")
    else:
        print(f"Building RevDeflate grader image {tag}...")
        subprocess.run(["docker", "build", "-t", tag, ctx], check=True)
        print(f"Built RevDeflate grader image {tag}")
    _revdeflate_grader_tag = tag
    return tag


def revdeflate_score(config, challenge_dir, submission_dir):
    """Grade recovered originals against the private ground truth in the grader
    image and read the 0-6 score it writes to /tmp/result.json. The submission is
    inert data (recovered files/trees under /submission/L1..L6); the grader only
    byte-compares it to the baked originals -- it never executes anything from the
    submission and never decompresses. Anti-smuggle: test.sh strips any symlinks
    from /submission before grading so a submission cannot alias the private
    originals. --network none.
    """
    image = _get_revdeflate_grader_image()
    submission_dir = os.path.abspath(submission_dir)

    container_name = f"revbench-revdeflate-{os.getpid()}"
    host_result = tempfile.NamedTemporaryFile(
        prefix="revbench-revdeflate-", suffix=".json", delete=False
    ).name
    cp_returncode = 1
    result = None
    try:
        subprocess.run(
            [
                "docker",
                "create",
                "--name",
                container_name,
                "--network",
                "none",
                "--security-opt",
                "no-new-privileges",
                "--cpus",
                "1",
                "--memory",
                "512m",
                image,
                "/grader/test.sh",
            ],
            check=True,
            capture_output=True,
        )
        # Stream the submission into the grader via a root-privileged tar (the grader
        # image runs as root) so root-owned, restrictive-mode dirs -- which revdeflate
        # legitimately grades on permission bits -- stay readable and keep their exact
        # modes, whatever the harness user's privileges. A host-side `docker cp` tars as
        # the harness user and fails on such trees on a non-root host.
        parent = os.path.dirname(submission_dir)
        base = os.path.basename(submission_dir)
        tar_proc = subprocess.Popen(
            ["docker", "run", "--rm", "-i", "--network", "none",
             "-v", f"{parent}:/w:ro", image,
             "tar", "-C", "/w", "--numeric-owner", "-cf", "-", base],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        cp_in = subprocess.run(
            ["docker", "cp", "-", f"{container_name}:/"],
            stdin=tar_proc.stdout,
            capture_output=True,
        )
        tar_proc.stdout.close()
        tar_proc.wait()
        if cp_in.returncode != 0:
            raise RuntimeError(
                "grader submission load failed: "
                + cp_in.stderr.decode(errors="replace")
            )
        result = subprocess.run(
            ["docker", "start", "-a", container_name],
            capture_output=True,
            text=True,
            timeout=240,
        )
        cp = subprocess.run(
            ["docker", "cp", f"{container_name}:/tmp/result.json", host_result],
            capture_output=True,
        )
        cp_returncode = cp.returncode
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    score = 0
    max_score = 6
    levels = {}
    if cp_returncode == 0:
        try:
            with open(host_result) as f:
                parsed = json.load(f)
            score = int(parsed.get("score", 0))
            max_score = int(parsed.get("max", 6))
            levels = parsed.get("levels") or {}
        except (ValueError, json.JSONDecodeError, OSError):
            pass
    if os.path.isfile(host_result):
        try:
            os.unlink(host_result)
        except OSError:
            pass

    return {
        "correct": score >= 1,
        "additional_info": {
            "score": score,
            "max": max_score,
            "levels": levels,
            "stdout": result.stdout.strip()[-2000:] if result and result.stdout else "",
            "stderr": result.stderr.strip()[-2000:] if result and result.stderr else "",
        },
    }
