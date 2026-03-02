---
name: engram-search
description: >-
  This skill should be used when the user asks to "search the codebase",
  "find related code", "how does X work", "what handles Y", "find where Z is implemented",
  "explore the architecture", "understand this module", or asks conceptual questions about
  code structure, patterns, or relationships. Also use when starting a new session to
  retrieve prior session context, or when finishing a session to save a summary.
  Prefer this skill over raw Grep/Glob for any semantic or exploratory code search —
  engram understands code meaning, not just text patterns.
---

# Engram Semantic Search

Engram is a local GPU-accelerated semantic code index running as an MCP server. It
maintains a live vector index of code chunks embedded via ONNX Runtime on CUDA, enabling
meaning-based code search rather than keyword matching.

## When to Use Engram vs Built-in Tools

The core decision is **semantic vs exact**:

| Use Engram (`mcp__engram__*`) | Use built-in Grep/Glob/Read |
|-------------------------------|------------------------------|
| "How does depth fusion work?" | `class DepthFusion` (exact name) |
| "Find code related to authentication" | `*.tsx` files in `src/components/` |
| "What handles error recovery?" | A specific string literal or regex |
| "Understand this module's architecture" | Reading a known file path |
| Exploratory questions about unfamiliar code | Directed lookup of a known symbol |

**Rule of thumb**: If the query is a natural-language question or conceptual exploration,
use engram. If it's an exact symbol name, file pattern, or text literal, use Grep/Glob.

## Tool Selection Guide

### 1. `search_code` — Semantic Search (Primary Tool)

The main workhorse. Embeds the query on GPU and returns ranked code chunks by cosine
similarity. Use for any natural-language question about code.

- **Required**: `query` (natural language)
- **Optional**: `limit` (default 10)
- Returns: ranked chunks with file path, line range, source text, similarity score

**When to use**: Conceptual questions, finding implementations by description, exploring
how features work, discovering related code across files.

### 2. `search_symbol` — Symbol Lookup

Finds functions, classes, or variables by name. Uses substring matching (case-insensitive),
not semantic similarity. Faster than `search_code` for known symbol names.

- **Required**: `name` (symbol name or substring)
- **Optional**: `kind` — `"function"`, `"class"`, or `"any"` (default)
- Returns: matching chunks with source text and location

**When to use**: Looking up a specific function, class, or struct by name when the exact
spelling is known but the file location isn't.

### 3. `get_context` — Contextual Expansion

Given a file and line number, retrieves surrounding code chunks and semantically related
chunks from elsewhere in the project.

- **Required**: `file` (relative path), `line` (1-based line number)
- **Optional**: `radius` (lines of context, default 50)
- Returns: local chunks within the radius + semantically related chunks from other files

**When to use**: After finding a result via `search_code` or `search_symbol`, expand
context to understand surrounding code. Also useful to discover cross-file relationships.

### 4. `get_session_memory` — Retrieve Past Session Context

Loads summaries from previous coding sessions. Optionally filters by a relevance query
using keyword matching (all query words must appear in the session text).

- **Optional**: `query` (relevance filter)
- Returns: session summaries with timestamps, key files, and key decisions

**When to use**: At the start of a session to recall what was done previously, or when
the user references past work ("remember when we fixed the MCP issue?").

### 5. `save_session_summary` — Persist Session Context

Saves a summary of the current session for future retrieval. Include what was accomplished,
which files were changed, and key design decisions.

- **Required**: `summary` (free-text description of what was accomplished)
- **Optional**: `key_files` (array of important file paths), `key_decisions` (array of
  design/implementation decisions)
- Returns: confirmation with session ID and timestamp

**When to use**: At the end of a productive session, especially when significant decisions
were made or multi-step work was done that a future session should know about.

## Recommended Workflow

### Starting a Session

1. Call `get_session_memory` (no query) to see recent sessions
2. If the user is continuing previous work, use `get_session_memory` with a relevant
   query to find the specific session context

### During Work

1. For conceptual questions → `search_code` first
2. For known symbols → `search_symbol` directly
3. To expand any result → `get_context` with the file and line from the result
4. Fall back to Grep only for exact text patterns that semantic search won't match

### Ending a Session

When the user indicates the session is ending, or significant work has been completed,
call `save_session_summary` with:
- A concise summary of what was accomplished
- The key files that were modified or created
- Any important design or implementation decisions

## Important Notes

- Engram indexes the project directory configured at startup. Chunks are relative to
  the project root.
- The index updates incrementally via a file watcher — edits, new files, and deletions
  are reflected automatically.
- `search_code` requires the GPU embedder to be running. If it returns an error about
  the embedder not being configured, fall back to `search_symbol` for name-based lookup
  or built-in Grep.
- Results include `score` fields (0.0-1.0 cosine similarity). Scores above 0.7 indicate
  strong relevance; below 0.4 may be noise.

## Additional Resources

For detailed parameter reference and usage examples, consult:
- **`references/tool-guide.md`** — Complete parameter reference, example queries, and
  decision flowchart for tool selection
