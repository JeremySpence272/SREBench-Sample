#!/usr/bin/env python3
"""Docker network isolation for RevBench containers.

Creates a custom Docker bridge network and uses iptables rules to restrict
outbound traffic to only the hosts required by the agent's API calls.
"""

import socket
import subprocess
import sys

NETWORK_NAME = "revbench-isolated"
IPTABLES_CHAIN = "DOCKER-USER"
# Comment marker used to identify our rules for cleanup
RULE_MARKER = "revbench-isolate"


def resolve_hosts(hosts):
    """Resolve hostnames to IP addresses. Returns a deduplicated set of IPs."""
    ips = set()
    for host in hosts:
        try:
            results = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
            for _, _, _, _, sockaddr in results:
                ips.add(sockaddr[0])
        except socket.gaierror:
            print(f"WARNING: could not resolve {host}, skipping")
    return ips


def _run(cmd, check=True):
    if len(cmd) > 1 and cmd[1] == "iptables":
        # -n: non-interactive sudo -- fail fast instead of hanging on a password
        #     prompt when the box lacks a passwordless iptables rule (needed so a
        #     background --no-internet sweep never blocks at setup/teardown).
        # -w 5: wait for the xtables lock so parallel --no-internet runs don't
        #     collide and error.
        cmd = [cmd[0], "-n", cmd[1], "-w", "5"] + cmd[2:]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_network_subnet():
    """Get the subnet of the revbench-isolated network."""
    result = _run(
        [
            "docker",
            "network",
            "inspect",
            NETWORK_NAME,
            "--format",
            "{{range .IPAM.Config}}{{.Subnet}}{{end}}",
        ]
    )
    return result.stdout.strip()


def create_network():
    """Create the isolated Docker bridge network if it doesn't exist."""
    result = subprocess.run(
        ["docker", "network", "inspect", NETWORK_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        return  # already exists

    print(f"Creating Docker network '{NETWORK_NAME}'...")
    r = subprocess.run(
        [
            "docker",
            "network",
            "create",
            "--driver",
            "bridge",
            "--opt",
            "com.docker.network.bridge.name=br-revbench",
            NETWORK_NAME,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        # A concurrent run may have created it between our inspect and create;
        # tolerate that (idempotent), but fail loudly on any other error.
        check = subprocess.run(
            ["docker", "network", "inspect", NETWORK_NAME], capture_output=True
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"failed to create network {NETWORK_NAME}: {r.stderr.strip()}"
            )


def setup_iptables(allowed_hosts):
    """Set up iptables rules to restrict container egress to allowed hosts only.

    Strategy:
      1. Resolve allowed_hosts to IPs
      2. In DOCKER-USER chain:
         - ACCEPT established/related (return traffic)
         - ACCEPT traffic from our subnet to each allowed IP on port 443
         - ACCEPT DNS to Docker's embedded DNS (127.0.0.11)
         - DROP all other FORWARD traffic from our subnet
    """
    subnet = get_network_subnet()
    if not subnet:
        print("ERROR: could not determine network subnet", file=sys.stderr)
        sys.exit(1)

    allowed_ips = resolve_hosts(allowed_hosts)
    if allowed_hosts and not allowed_ips:
        print("ERROR: no IPs resolved for allowed hosts", file=sys.stderr)
        sys.exit(1)

    print(f"Network subnet: {subnet}")
    print(f"Allowing {len(allowed_ips)} IPs for {len(allowed_hosts)} hosts")

    # Clean any existing revbench rules first
    cleanup_iptables()

    # Allow established/related connections (for return traffic)
    _run(
        [
            "sudo",
            "iptables",
            "-I",
            IPTABLES_CHAIN,
            "-s",
            subnet,
            "-m",
            "conntrack",
            "--ctstate",
            "ESTABLISHED,RELATED",
            "-m",
            "comment",
            "--comment",
            RULE_MARKER,
            "-j",
            "ACCEPT",
        ]
    )

    # Allow DNS to Docker's embedded DNS server (UDP and TCP port 53)
    for proto in ("udp", "tcp"):
        _run(
            [
                "sudo",
                "iptables",
                "-I",
                IPTABLES_CHAIN,
                "-s",
                subnet,
                "-d",
                "127.0.0.11/32",
                "-p",
                proto,
                "--dport",
                "53",
                "-m",
                "comment",
                "--comment",
                RULE_MARKER,
                "-j",
                "ACCEPT",
            ]
        )

    # Allow HTTPS to each resolved IP
    for ip in sorted(allowed_ips):
        _run(
            [
                "sudo",
                "iptables",
                "-I",
                IPTABLES_CHAIN,
                "-s",
                subnet,
                "-d",
                ip,
                "-p",
                "tcp",
                "--dport",
                "443",
                "-m",
                "comment",
                "--comment",
                RULE_MARKER,
                "-j",
                "ACCEPT",
            ]
        )

    # DROP everything else from our subnet (append so it comes after the ACCEPTs)
    _run(
        [
            "sudo",
            "iptables",
            "-A",
            IPTABLES_CHAIN,
            "-s",
            subnet,
            "-j",
            "DROP",
            "-m",
            "comment",
            "--comment",
            RULE_MARKER,
        ]
    )

    print("iptables rules installed for network isolation.")


def cleanup_iptables():
    """Remove all iptables rules tagged with our marker comment."""
    # List rules with line numbers, then delete in reverse order
    result = _run(
        ["sudo", "iptables", "-L", IPTABLES_CHAIN, "--line-numbers", "-n"],
        check=False,
    )
    if result.returncode != 0:
        return

    # Find lines containing our marker and extract rule numbers
    rule_nums = []
    for line in result.stdout.splitlines():
        if RULE_MARKER in line:
            parts = line.split()
            if parts and parts[0].isdigit():
                rule_nums.append(int(parts[0]))

    # Delete in reverse order to preserve numbering
    for num in sorted(rule_nums, reverse=True):
        _run(
            ["sudo", "iptables", "-D", IPTABLES_CHAIN, str(num)],
            check=False,
        )

    if rule_nums:
        print(f"Cleaned up {len(rule_nums)} iptables rules.")


def remove_network():
    """Remove the Docker network (only succeeds if no containers are attached)."""
    subprocess.run(
        ["docker", "network", "rm", NETWORK_NAME],
        capture_output=True,
    )


def setup_isolation(allowed_hosts):
    """Full setup: create network and configure iptables rules."""
    create_network()
    setup_iptables(allowed_hosts)


def teardown_isolation():
    """Full teardown: remove iptables rules and Docker network."""
    cleanup_iptables()
    remove_network()
