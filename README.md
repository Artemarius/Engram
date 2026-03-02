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
- NVIDIA GPU with CUDA 12.x + cuDNN 9.x (for GPU-accelerated embedding; CPU fallback available)
- ONNX Runtime 1.24+ GPU package (optional, for semantic search)
- Claude Code with MCP support

## Tech Stack

- **C++17** — core indexing engine and MCP server
- **CUDA / ONNX Runtime** — GPU-accelerated embedding inference
- **hnswlib** — HNSW vector index (header-only C++)
- **tree-sitter** — AST-aware code chunking (9 languages: C++, Python, JS, TS, Java, Rust, Go, Ruby, C#)
- **nlohmann/json** — JSON-RPC message handling
- **spdlog** — structured logging (stderr only)
- **Google Test** — unit testing
- **Python** — model export and validation scripts

## Building

```bash
# Core build (no GPU embedding — symbol search and session memory still work)
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure

# With tree-sitter for AST-aware chunking (more accurate symbol boundaries)
cmake -B build -G "Visual Studio 17 2022" -A x64 \
  -DENGRAM_USE_TREESITTER=ON
cmake --build build --config Release

# With ONNX Runtime for GPU-accelerated semantic search
cmake -B build -G "Visual Studio 17 2022" -A x64 \
  -DENGRAM_USE_ONNX=ON \
  -DONNXRUNTIME_ROOT="path/to/onnxruntime-win-x64-gpu"
cmake --build build --config Release

# Full build (tree-sitter + ONNX)
cmake -B build -G "Visual Studio 17 2022" -A x64 \
  -DENGRAM_USE_TREESITTER=ON \
  -DENGRAM_USE_ONNX=ON \
  -DONNXRUNTIME_ROOT="path/to/onnxruntime-win-x64-gpu"
cmake --build build --config Release
```

## Model Export

Before running with GPU embedding, export the ONNX model:

```bash
# Create a Python venv (once)
python -m venv .venv
.venv/Scripts/activate  # Windows

# Install dependencies
pip install torch transformers onnx onnxruntime-gpu

# Export all-MiniLM-L6-v2 (384-dim, ~86MB) — recommended for 6GB VRAM GPUs
python scripts/export_model.py --model minilm --output models/ --validate

# Or export with INT8 quantization for smaller size
python scripts/export_model.py --model minilm --output models/ --quantize --validate
```

This writes `models/all-MiniLM-L6-v2.onnx` and `models/tokenizer.json`.

## Usage

```bash
# Run the MCP server directly (for testing)
./build/bin/engram-mcp.exe --project /path/to/your/repo --model models/all-MiniLM-L6-v2.onnx

# Register with Claude Code (user scope — available in all projects)
claude mcp add engram --scope user -- \
  /path/to/engram-mcp.exe \
  --project /path/to/your/repo \
  --model /path/to/models/all-MiniLM-L6-v2.onnx

# Force re-index on next connection (use after code changes while server was offline)
claude mcp remove engram --scope user
claude mcp add engram --scope user -- \
  /path/to/engram-mcp.exe \
  --project /path/to/your/repo \
  --model /path/to/models/all-MiniLM-L6-v2.onnx \
  --reindex

# Claude Code will automatically discover the tools in a new session.
# The bundled skill teaches Claude when to use engram vs built-in search.
# Just ask naturally: "how is camera calibration implemented?"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--project <path>` | Root of the codebase to index |
| `--model <path>` | Path to the ONNX embedding model |
| `--data-dir <path>` | Directory for persistent data (default: `<project>/.engram/`) |
| `--dim <int>` | Embedding dimension (default: 384, auto-detected from model) |
| `--reindex` | Force a full re-index of the project |
| `--treesitter` | Use tree-sitter AST-aware chunker (requires `ENGRAM_USE_TREESITTER` build) |
| `--verbose` | Enable debug-level logging |

## License

MIT
