#!/bin/bash
# Hook: SessionStart (fires on startup and resume)
# Reminds Claude to retrieve prior session context from engram.

cat <<'EOF'
You are working in the Engram project which has an MCP-based session memory system.
Please call the `get_session_memory` engram tool now to retrieve relevant context
from previous coding sessions. This helps maintain continuity across sessions.
EOF
