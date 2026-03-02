# Engram Tool Reference

Complete parameter reference, example queries, and decision guidance for each engram
MCP tool.

## Table of Contents

1. [search_code](#search_code)
2. [search_symbol](#search_symbol)
3. [get_context](#get_context)
4. [get_session_memory](#get_session_memory)
5. [save_session_summary](#save_session_summary)
6. [Decision Flowchart](#decision-flowchart)

---

## search_code

Semantic code search using GPU-accelerated embeddings. The query is embedded via the
same model used to index the codebase, and results are ranked by cosine similarity.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Natural-language search query |
| `limit` | integer | No | 10 | Maximum number of results to return |

### Response Fields

Each result contains:
- `file_path` — relative path from project root
- `start_line`, `end_line` — line range of the chunk
- `language` — detected language (cpp, python, js, etc.)
- `source_text` — the actual code
- `symbol_name` — function/class name if detected
- `chunk_id` — internal identifier
- `score` — cosine similarity (0.0-1.0; higher = more relevant)

### Example Queries

**Broad conceptual search:**
```
query: "error handling and recovery logic"
limit: 10
```

**Implementation-specific search:**
```
query: "HNSW vector index persistence and loading"
limit: 5
```

**Cross-cutting concern:**
```
query: "thread safety and mutex usage"
limit: 8
```

### Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 0.75 - 1.0 | Strong match — highly relevant code |
| 0.50 - 0.75 | Moderate match — related but possibly tangential |
| 0.30 - 0.50 | Weak match — may contain some relevance |
| < 0.30 | Noise — unlikely to be useful |

### Tips

- Phrase queries as descriptions of what the code does, not what it's called
- Longer, more specific queries tend to produce better results than single words
- If results seem off-topic, try rephrasing the query with domain-specific terms
- Use `limit: 5` for focused questions, `limit: 15-20` for broad exploration

---

## search_symbol

Name-based symbol lookup. Scans all indexed chunks for case-insensitive substring
matches on the `symbol_name` field. Does not use embeddings — fast for exact lookups.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | — | Symbol name or substring to search for |
| `kind` | string | No | `"any"` | Filter: `"function"`, `"class"`, or `"any"` |

### Response Fields

Same as `search_code` results, minus the `score` field.

### Example Queries

**Find a specific class:**
```
name: "HnswIndex"
kind: "class"
```

**Find functions containing a word:**
```
name: "embed"
kind: "function"
```

**Find anything matching a pattern:**
```
name: "session"
kind: "any"
```

### Kind Filtering

The `kind` filter uses heuristic detection (not AST-based):
- `"function"` — looks for `def `, `function `, `fn `, or `name(` patterns
- `"class"` — looks for `class ` or `struct ` patterns
- `"any"` — returns all symbol name matches regardless of type

Since detection is heuristic, prefer `kind: "any"` when uncertain and filter results
manually if needed.

### Tips

- Substring matching means `name: "chunk"` matches `ChunkStore`, `chunk_to_json`,
  `regex_chunker`, etc.
- Use `search_symbol` when the exact name is known; use `search_code` when describing
  what the code does
- Combine with `get_context` to expand results into their surrounding code

---

## get_context

Retrieves code chunks around a specific file and line, plus semantically related code
from other files. Useful for understanding how a piece of code fits into the broader
codebase.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | string | Yes | — | File path (relative to project root) |
| `line` | integer | Yes | — | Line number (1-based) |
| `radius` | integer | No | 50 | Number of surrounding lines to include |

### Response Fields

- `results` — chunks from the specified file within the radius
- `related` — semantically similar chunks from other files (if embedder is available)
- `related_count` — number of related chunks found

### File Path Matching

The tool tries multiple matching strategies:
1. Exact absolute path match
2. Relative path from project root
3. Filename-based match (suffix match on relative path)

Relative paths work best: `src/mcp/tools.cpp` rather than full absolute paths.

### Example Usage

**Expand a search result:**
After `search_code` returns a chunk at `src/index/hnsw_index.cpp:45`:
```
file: "src/index/hnsw_index.cpp"
line: 45
radius: 50
```

**Understand a function in context:**
```
file: "src/mcp/mcp_server.cpp"
line: 120
radius: 100
```

### Tips

- The `related` field is the most valuable part — it shows cross-file relationships
  that text search cannot find
- Use after `search_code` or `search_symbol` to expand results
- Increase `radius` to 100+ for large functions or class definitions
- Results from `related` have similarity scores; use these to gauge relevance

---

## get_session_memory

Retrieves summaries from previous coding sessions. Uses word-level keyword matching when
a query is provided (all words in the query must appear somewhere in the session text).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | No | `""` | Relevance query to filter sessions |

### Response Fields

Each session contains:
- `id` — session identifier (timestamp-based: YYYYMMDD_HHMMSS)
- `timestamp` — human-readable timestamp
- `summary` — free-text summary of the session
- `key_files` — array of important files from that session
- `key_decisions` — array of design/implementation decisions

### Example Usage

**Retrieve all recent sessions:**
```
(no parameters — returns all sessions, most recent first)
```

**Filter by topic:**
```
query: "MCP protocol"
```
This matches any session whose combined text contains both "MCP" and "protocol"
(case-insensitive, words can appear anywhere).

### Tips

- Call with no query at session start to see recent context
- Use specific multi-word queries to narrow down to relevant sessions
- Sessions are sorted most-recent-first
- The keyword matching is word-level: "MCP protocol" matches "MCP stdio protocol"
  but not "MCP" alone

---

## save_session_summary

Persists a summary of the current coding session to disk. Saved sessions can be
retrieved in future sessions via `get_session_memory`.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `summary` | string | Yes | — | Free-text summary of what was accomplished |
| `key_files` | string[] | No | `[]` | Important files touched in this session |
| `key_decisions` | string[] | No | `[]` | Key design or implementation decisions |

### Response Fields

- `status` — `"saved"` on success
- `session_id` — generated ID (YYYYMMDD_HHMMSS format)
- `timestamp` — human-readable save time

### Writing Good Summaries

A useful session summary should capture:

1. **What was accomplished** — the concrete outcome, not the process
2. **Key files** — files that were created, modified, or are important for context
3. **Key decisions** — design choices, trade-offs, rejected alternatives

**Good example:**
```json
{
  "summary": "Implemented tree-sitter chunker for C++ and Python. Replaced regex-based function detection with AST-aware parsing. Added 35 new tests.",
  "key_files": [
    "src/chunker/treesitter_chunker.cpp",
    "src/chunker/treesitter_chunker.hpp",
    "tests/test_treesitter_chunker.cpp"
  ],
  "key_decisions": [
    "Used tree-sitter C bindings via FetchContent rather than vendoring",
    "Kept regex chunker as fallback for unsupported languages",
    "Chunk boundaries at function/class level, not statement level"
  ]
}
```

### Tips

- Save at the end of productive sessions, not trivial ones
- Focus on decisions and outcomes that a future session would need to know
- Include file paths so `get_session_memory` results are immediately actionable

---

## Decision Flowchart

```
User asks a question about code
│
├─ Is it about a specific symbol name?
│  ├─ Yes → search_symbol(name, kind)
│  └─ No ↓
│
├─ Is it a conceptual/exploratory question?
│  ├─ Yes → search_code(query)
│  └─ No ↓
│
├─ Is it about a specific file and location?
│  ├─ Yes → get_context(file, line)
│  └─ No ↓
│
├─ Is it an exact text pattern or regex?
│  ├─ Yes → Use built-in Grep
│  └─ No ↓
│
├─ Is it about file names or paths?
│  ├─ Yes → Use built-in Glob
│  └─ No ↓
│
└─ Is it about past work or session history?
   ├─ Yes → get_session_memory(query)
   └─ No → Use Read to examine specific files
```

### Common Patterns

**Explore unfamiliar code:**
1. `search_code` with a broad query
2. `get_context` on the most relevant result
3. Read specific files if needed

**Find and understand a function:**
1. `search_symbol` to locate it
2. `get_context` to see surrounding code and cross-file relationships

**Continue previous work:**
1. `get_session_memory` to recall context
2. `search_code` or `search_symbol` to re-find the relevant code

**End-of-session:**
1. `save_session_summary` with what was done, files touched, decisions made
