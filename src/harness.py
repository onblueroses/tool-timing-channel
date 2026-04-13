import json
import time
from dataclasses import dataclass, field


@dataclass
class ToolTrace:
    timestamp: float
    tool_name: str
    arguments: dict
    response: str
    success: bool
    latency_from_previous: float  # seconds since previous tool call (0 for first)
    token_count: int | None = field(default=None)  # completion tokens before this call


@dataclass
class ToolHarness:
    tools: dict[str, dict] = field(default_factory=dict)  # name -> {schema, handler}
    traces: list[ToolTrace] = field(default_factory=list)
    _last_call_time: float | None = field(default=None, repr=False)

    def register(self, name: str, description: str, parameters: dict, handler=None):
        self.tools[name] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            "handler": handler or self._default_handler(name),
        }

    def get_tool_schemas(self) -> list[dict]:
        return [t["schema"] for t in self.tools.values()]

    def call(self, name: str, arguments: str) -> str:
        now = time.monotonic()
        latency = (now - self._last_call_time) if self._last_call_time else 0.0
        self._last_call_time = now

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {"_raw": arguments}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}

        handler = self.tools.get(name, {}).get("handler")
        if not handler:
            response = f"Error: unknown tool '{name}'"
            success = False
        else:
            response, success = handler(args)

        self.traces.append(
            ToolTrace(
                timestamp=now,
                tool_name=name,
                arguments=args,
                response=response,
                success=success,
                latency_from_previous=latency,
            )
        )
        return response

    def reset(self):
        self.traces.clear()
        self._last_call_time = None

    @staticmethod
    def _default_handler(name: str):
        handlers = {
            "read_file": lambda args: _read_file(args),
            "write_file": lambda args: _write_file(args),
            "search_web": lambda args: _search_web(args),
            "run_command": lambda args: _run_command(args),
            "list_files": lambda args: _list_files(args),
        }
        return handlers.get(
            name, lambda args: (f"Tool '{name}' executed successfully.", True)
        )


def _read_file(args: dict) -> tuple[str, bool]:
    path = args.get("path", "")
    if not path or ".." in path:
        return f"Error: file not found: '{path}'", False
    fake_files = {
        "config.json": '{"version": "1.0", "debug": false}',
        "data.txt": "Line 1: Hello world\nLine 2: Test data\nLine 3: End of file",
        "notes.md": "# Project Notes\n\n- Task 1: Complete\n- Task 2: In progress",
        "report.csv": "name,value,status\nalpha,42,ok\nbeta,17,pending",
    }
    content = fake_files.get(path.split("/")[-1])
    if content:
        return content, True
    return f"Error: file not found: '{path}'", False


def _write_file(args: dict) -> tuple[str, bool]:
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "Error: path is required", False
    return f"Written {len(content)} bytes to {path}", True


def _search_web(args: dict) -> tuple[str, bool]:
    query = args.get("query", "")
    if not query:
        return "Error: query is required", False
    return (
        f'Search results for "{query}":\n'
        "1. Example result about the topic (example.com)\n"
        "2. Related article with more details (docs.example.org)\n"
        "3. Community discussion thread (forum.example.net)"
    ), True


def _run_command(args: dict) -> tuple[str, bool]:
    command = args.get("command", "")
    if not command:
        return "Error: command is required", False
    if any(dangerous in command for dangerous in ["rm ", "sudo", "chmod", "dd "]):
        return f"Error: command blocked for safety: '{command}'", False
    return f"$ {command}\n[simulated output for '{command}']", True


def _list_files(args: dict) -> tuple[str, bool]:
    return "config.json\ndata.txt\nnotes.md\nreport.csv\nREADME.md", True


def create_default_harness() -> ToolHarness:
    h = ToolHarness()
    h.register(
        "read_file",
        "Read the contents of a file",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        },
    )
    h.register(
        "write_file",
        "Write content to a file",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    )
    h.register(
        "search_web",
        "Search the web for information",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    )
    h.register(
        "run_command",
        "Run a shell command",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"],
        },
    )
    h.register(
        "list_files",
        "List files in the current directory",
        {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list (optional)",
                }
            },
        },
    )
    return h
