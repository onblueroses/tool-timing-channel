# Security Policy

## Purpose of This Research

This repository demonstrates covert side channels in AI agent tool calls. It exists to help the community understand and defend against these threats, not to enable attacks.

## Responsible Use

This code is designed for:
- Security research and analysis
- Building detection tools for agent frameworks
- Designing safer agent architectures
- Educational purposes

This code should NOT be used for:
- Attacking production systems
- Exfiltrating real data from deployed agents
- Bypassing safety measures in commercial AI products

## Safe by Construction

All experiments use a **fake tool harness** that returns canned responses. No real filesystem, network, or system access occurs during experiments. The harness is intentionally limited - it cannot be repurposed for real attacks without significant modification.

## Reporting Issues

If you discover that these techniques work against a specific production agent framework, please report it to the framework vendor through their responsible disclosure process before publishing.

For issues with this repository itself, open a GitHub issue.

## Threat Model

See [analysis/FINDINGS.md](analysis/FINDINGS.md) for the full threat model, including attacker capabilities needed, existing defenses, and proposed mitigations.
