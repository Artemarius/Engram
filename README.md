# Engram

GPU-accelerated local semantic code index for Claude Code. Embeds your codebase using CUDA-optimized inference, maintains a live vector index with incremental updates, and serves precise context via MCP server. Replaces brute-force file reads with intelligent retrieval.

## Motivation

Claude Code reads entire files to find relevant context — a function signature here, a type definition there. On a large codebase, this burns through the context window fast. By the time you've explained the problem and Claude has loaded the relevant files, half your budget is gone.

Engram fixes this by maintaining a persistent semantic index of your codebase locally. Instead of reading 10 files to find the 3 functions that matter, Claude Code queries Engram and gets back precisely the relevant code snippets — with file paths, line numbers, and similarity scores.

## How It Works

1. **File watcher** monitors your project for changes (new files, edits, deletions)
2. **Chunker** splits code into semantic units (functions, classes, blocks) using language-aware parsing
3. **Embedding engine** runs a quantized code embedding model on your GPU via ONNX Runtime + CUDA
4. **Vector index** stores embeddings in an HNSW graph for fast approximate nearest-neighbor search
5. **MCP server** exposes search tools over stdio transport — Claude Code queries naturally

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  File Watcher │────>│   Chunker    │────>│  Embedder    │
│  (filesystem) │     │  (tree-sitter│     │  (ONNX+CUDA) │
└──────────────┘     │   or regex)  │     └──────┬───────┘
                     └──────────────┘            │
                                                 v
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Claude Code │<────│  MCP Server  │<────│ Vector Index │
│  (queries)   │     │  (stdio)     │     │  (HNSW)      │
└──────────────┘     └──────────────┘     └──────────────┘
```

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search: "how is depth fusion implemented" → ranked snippets |
| `search_symbol` | Find by symbol name: function, class, struct |
| `get_context` | Given a file and line range, retrieve related code across the project |
| `get_session_memory` | Retrieve summaries from previous coding sessions |
| `save_session_summary` | Persist key decisions/changes from current session |

## Requirements

- Windows 10/11
- CMake 3.24+
- Visual Studio 2022 (MSVC)
- NVIDIA GPU with CUDA 12.x (for GPU-accelerated embedding; CPU fallback available)
- Claude Code with MCP support

## Tech Stack

- **C++17** — core indexing engine and MCP server
- **CUDA / ONNX Runtime** — GPU-accelerated embedding inference
- **hnswlib** — HNSW vector index (header-only C++)
- **tree-sitter** — language-aware code chunking (planned)
- **nlohmann/json** — JSON-RPC message handling
- **spdlog** — structured logging (stderr only)
- **Google Test** — unit testing
- **Python** — model export and validation scripts

## Building

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

## Usage

```bash
# Register with Claude Code
claude mcp add engram -- ./build/bin/engram-mcp.exe --project /path/to/your/repo

# Claude Code will automatically discover the tools
# Try: "search for code related to camera calibration"
```

## License

MIT
