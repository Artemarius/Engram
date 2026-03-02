"""Minimal MCP server to test if Claude Code can connect at all."""
import sys
import json

def read_message():
    """Read a newline-delimited JSON message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)

def send_response(obj):
    """Send a newline-delimited JSON response to stdout."""
    body = json.dumps(obj)
    sys.stdout.write(body + "\n")
    sys.stdout.flush()

def main():
    sys.stderr.write("mcp_test_server: starting\n")
    sys.stderr.flush()

    while True:
        msg = read_message()
        if msg is None:
            sys.stderr.write("mcp_test_server: stdin EOF\n")
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")
        sys.stderr.write(f"mcp_test_server: received {method} (id={msg_id})\n")
        sys.stderr.flush()

        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "engram-test", "version": "0.1.0"}
                }
            })
            sys.stderr.write("mcp_test_server: sent initialize response\n")
            sys.stderr.flush()

        elif method == "notifications/initialized":
            sys.stderr.write("mcp_test_server: client initialized\n")
            sys.stderr.flush()
            # No response for notifications

        elif method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": []}
            })
            sys.stderr.write("mcp_test_server: sent tools/list response\n")
            sys.stderr.flush()

        else:
            sys.stderr.write(f"mcp_test_server: unknown method {method}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    main()
