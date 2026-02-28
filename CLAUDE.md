# CLAUDE.md — Engram

## What This Project Is

A local GPU-accelerated semantic code index that serves as an MCP server for Claude Code. It embeds code chunks using a quantized model on CUDA, stores them in an HNSW vector index, and exposes intelligent code search tools via the MCP stdio protocol. The goal is to replace brute-force file reads with precise, relevant context retrieval.

**Read PROJECT.md for full strategic context and motivation.**

## Developer Background

Artem has deep expertise in C++ (15+ years), CUDA, computer vision, 3D reconstruction, and GPU optimization. This project bridges his systems programming background with ML inference deployment and developer tooling. He's learning MCP protocol and embedding model deployment through this project.

## Development Environment

- **OS**: Windows 10 Pro 22H2
- **Compiler**: MSVC (Visual Studio 2022)
- **GPU**: RTX 3060 6GB VRAM (compute capability 8.6)
- **CUDA**: 12.8
- **C++ Standard**: C++17
- **Build**: CMake 3.24+ with VS generator

## Architecture

```
engram/
├── CMakeLists.txt
├── CLAUDE.md
├── README.md
├── external/                  # FetchContent-managed dependencies
├── models/                    # ONNX model files (gitignored, downloaded separately)
│   └── .gitkeep
├── scripts/
│   ├── export_model.py        # Export embedding model to ONNX
│   └── test_embeddings.py     # Validate ONNX output vs PyTorch
├── src/
│   ├── chunker/               # Code splitting into semantic units
│   │   ├── chunker.hpp        # Abstract chunker interface
│   │   ├── treesitter_chunker.cpp/.hpp  # Tree-sitter based (preferred)
│   │   └── regex_chunker.cpp/.hpp       # Fallback regex-based
│   ├── embedder/              # ONNX Runtime inference
│   │   ├── embedder.hpp       # Interface
│   │   ├── ort_embedder.cpp/.hpp  # ONNX Runtime + CUDA EP
│   │   └── tokenizer.cpp/.hpp     # Tokenizer (from tokenizers-cpp or custom)
│   ├── index/                 # Vector storage and search
│   │   ├── vector_index.hpp   # Interface
│   │   └── hnsw_index.cpp/.hpp  # hnswlib wrapper with persistence
│   ├── watcher/               # Filesystem monitoring
│   │   ├── watcher.hpp
│   │   └── win_watcher.cpp/.hpp  # ReadDirectoryChangesW implementation
│   ├── mcp/                   # MCP protocol and tools
│   │   ├── mcp_server.cpp/.hpp   # JSON-RPC over stdio
│   │   ├── tools.cpp/.hpp        # Tool definitions and handlers
│   │   └── protocol.hpp          # MCP message types
│   ├── session/               # Session memory management
│   │   ├── session_store.cpp/.hpp  # Save/load session summaries
│   │   └── session_embedder.cpp/.hpp  # Embed and index session data
│   └── main.cpp               # Entry point, wires everything together
├── tests/
│   ├── test_chunker.cpp
│   ├── test_embedder.cpp
│   ├── test_index.cpp
│   └── test_mcp_protocol.cpp
└── data/                      # Persistent index data (gitignored)
    └── .gitkeep
```

## Key Technical Decisions

### Embedding Model
- Use a small code-optimized model: Nomic Embed Code, or `all-MiniLM-L6-v2` as fallback (384 dim, ~80MB ONNX)
- Export to ONNX FP16, optionally INT8 quantized
- Run via ONNX Runtime C++ API with CUDA Execution Provider
- Keep model files out of git — download script in `scripts/`

### Vector Index
- hnswlib (header-only C++, no dependencies)
- Cosine similarity space
- Persist index to disk, reload on startup
- Incremental: add/remove embeddings as files change
- Parameters: M=16, efConstruction=200, efSearch=50 (tune later)

### Code Chunking Strategy
- Prefer tree-sitter for language-aware parsing (functions, classes, methods)
- Fallback to regex-based splitting for unsupported languages
- Each chunk: source text, file path, line range, language, symbol name if available
- Target chunk size: 50-500 tokens (configurable)
- Store chunk metadata alongside embedding in the index

### MCP Protocol
- Communicate over stdio (stdin/stdout) using JSON-RPC 2.0
- NEVER write anything to stdout except MCP protocol messages
- All logging goes to stderr or file
- Implement `tools/list` and `tools/call` handlers
- Tool responses return code snippets with file paths and line numbers

### Session Memory
- On session end, accept a summary string from Claude Code
- Embed the summary and store in a separate "session" index
- On session start, retrieve relevant past session context
- Store as JSON: { timestamp, summary, key_files, key_decisions }

## Coding Conventions

- Use `std::filesystem` for all path operations
- Use `spdlog` for logging (stderr sink only)
- Use `nlohmann/json` for all JSON
- No exceptions in hot paths — use `std::expected` or error codes
- Prefix CUDA/GPU-specific code clearly
- All public APIs get doc comments
- Tests use Google Test

## Dependencies (all via FetchContent or vendored)

| Dependency | Purpose | Source |
|------------|---------|--------|
| hnswlib | Vector index | FetchContent (GitHub) |
| nlohmann/json | JSON parsing | FetchContent |
| spdlog | Logging | FetchContent |
| ONNX Runtime | ML inference | Pre-built CUDA package |
| tree-sitter | Code parsing | FetchContent |
| tree-sitter-cpp | C++ grammar | FetchContent |
| tree-sitter-python | Python grammar | FetchContent |
| Google Test | Testing | FetchContent |

## Build Commands

```bash
# Configure (first time)
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build
cmake --build build --config Release

# Run tests
cd build && ctest -C Release --output-on-failure

# Run the MCP server (for testing)
./build/Release/engram-mcp.exe --project . --model models/nomic-embed-code.onnx
```

## Things NOT to Do

- Don't write to stdout (reserved for MCP protocol)
- Don't use platform-specific APIs outside of `watcher/` module
- Don't heap-allocate in the search hot path
- Don't load the full ONNX model per query — keep session alive
- Don't store model files in git
- Don't use `std::map` for the chunk metadata store — use flat vectors or hash maps
