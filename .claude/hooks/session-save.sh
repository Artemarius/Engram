#!/bin/bash
# Hook: Stop (fires when Claude is about to end the session)
# Asks Claude to save a session summary to engram before stopping.
#
# Uses a guard variable to prevent infinite loops: the hook tells Claude
# to save, Claude calls save_session_summary, then stops again — the
# second stop must not re-trigger.

if [ -n "$stop_hook_active" ]; then
    exit 0
fi

export stop_hook_active=1

cat <<'EOF'
Before ending this session, please call the `save_session_summary` engram tool
to persist a summary of what was accomplished, key files touched, and any
important decisions made. This helps future sessions pick up where you left off.

After saving, you may proceed to stop.
EOF
