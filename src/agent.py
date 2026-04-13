import json
import re
import time
from dataclasses import dataclass

from .client import CompletionResult, chat_completion, get_client, DEFAULT_MODEL
from .harness import ToolHarness, ToolTrace, create_default_harness


PSEUDO_TOOL_CALL_RE = re.compile(
    r"(?m)^\s*(read_file|write_file|search_web|run_command|list_files)\((\{.*\})\)\s*$"
)
PSEUDO_TOOL_OBJECT_RE = re.compile(
    r"(?:^|.*?)(?:\()?\s*(read_file|write_file|search_web|run_command|list_files)\s+(\{.*\})\s*\)?$"
)


def _parse_pseudo_tool_calls(content: str | None) -> list[dict]:
    if not content:
        return []

    tool_calls = []
    lines = content.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        normalized = stripped
        while normalized.startswith("(") and normalized.endswith(")"):
            inner = normalized[1:-1].strip()
            if not inner:
                break
            normalized = inner

        match = PSEUDO_TOOL_CALL_RE.match(normalized)
        if match:
            tool_calls.append(
                {
                    "id": f"pseudo-call-{len(tool_calls) + 1}",
                    "name": match.group(1),
                    "arguments": match.group(2),
                }
            )
            continue

        if normalized.startswith("ReadOnly "):
            arguments = normalized.removeprefix("ReadOnly ").strip()
            tool_calls.append(
                {
                    "id": f"pseudo-call-{len(tool_calls) + 1}",
                    "name": "read_file",
                    "arguments": arguments,
                }
            )
            continue

        if normalized.startswith("hydrate(") and normalized.endswith(")"):
            payload_text = normalized[len("hydrate(") : -1]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                name = payload.get("name")
                arguments = payload.get("arguments")
                if isinstance(name, str):
                    tool_calls.append(
                        {
                            "id": f"pseudo-call-{len(tool_calls) + 1}",
                            "name": name,
                            "arguments": json.dumps(arguments or {}),
                        }
                    )
                    continue

        object_match = PSEUDO_TOOL_OBJECT_RE.match(normalized)
        if object_match:
            tool_calls.append(
                {
                    "id": f"pseudo-call-{len(tool_calls) + 1}",
                    "name": object_match.group(1),
                    "arguments": object_match.group(2),
                }
            )
            continue

        json_start = normalized.find('{"name"')
        if json_start == -1 and normalized.startswith("{"):
            json_start = 0
        if json_start != -1 and normalized.endswith("}"):
            try:
                payload = json.loads(normalized[json_start:])
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            arguments = payload.get("arguments")
            if isinstance(name, str) and name in {
                "read_file",
                "write_file",
                "search_web",
                "run_command",
                "list_files",
            }:
                tool_calls.append(
                    {
                        "id": f"pseudo-call-{len(tool_calls) + 1}",
                        "name": name,
                        "arguments": json.dumps(arguments or {}),
                    }
                )
    return tool_calls


def _normalize_tool_arguments(arguments) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return json.dumps({"_raw": arguments})
        return json.dumps(parsed)

    return json.dumps(arguments)


@dataclass
class AgentRun:
    system_prompt: str
    user_message: str
    model: str
    traces: list[ToolTrace]
    messages: list[dict]
    completions: list[CompletionResult]
    total_time: float
    iterations: int


def run_agent(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    harness: ToolHarness | None = None,
    max_iterations: int = 15,
) -> AgentRun:
    client = get_client(model)
    harness = harness or create_default_harness()
    harness.reset()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    completions = []
    start = time.monotonic()

    for i in range(max_iterations):
        result = chat_completion(
            client, messages, tools=harness.get_tool_schemas(), model=model
        )
        if not result.tool_calls:
            result.tool_calls = _parse_pseudo_tool_calls(result.content)
        completions.append(result)

        if not result.tool_calls:
            break

        # Build assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": result.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": _normalize_tool_arguments(tc["arguments"]),
                },
            }
            for tc in result.tool_calls
        ]
        messages.append(assistant_msg)

        # Execute each tool call and add results
        completion_tokens = result.usage.get("completion_tokens", 0)
        for tc in result.tool_calls:
            tool_response = harness.call(
                tc["name"], _normalize_tool_arguments(tc["arguments"])
            )
            # Attach completion token count to the trace for timing analysis
            if harness.traces and completion_tokens:
                harness.traces[-1].token_count = completion_tokens
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_response,
                }
            )

    elapsed = time.monotonic() - start

    return AgentRun(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        traces=list(harness.traces),
        messages=messages,
        completions=completions,
        total_time=elapsed,
        iterations=len(completions),
    )
