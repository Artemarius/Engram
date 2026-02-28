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

Files marked with `(*)` are interfaces only — implementation is planned for later phases.

```
engram/
├── CMakeLists.txt             # Build system with FetchContent deps
├── CLAUDE.md
├── README.md
├── models/                    # ONNX model files (gitignored, downloaded separately)
│   └── .gitkeep
├── scripts/
│   ├── export_model.py        # Export embedding model to ONNX
│   └── test_embeddings.py     # Validate ONNX output vs PyTorch
├── src/
│   ├── chunker/               # Code splitting into semantic units
│   │   ├── chunker.hpp        # Abstract chunker interface + Chunk struct
│   │   ├── regex_chunker.hpp  # Regex-based chunker (implemented)
│   │   └── regex_chunker.cpp
│   ├── embedder/              # ONNX Runtime inference
│   │   ├── embedder.hpp       # Abstract embedder interface (*)
│   │   └── tokenizer.hpp      # Abstract tokenizer interface (*)
│   ├── index/                 # Vector storage and search
│   │   ├── vector_index.hpp   # Abstract index interface
│   │   ├── hnsw_index.hpp     # hnswlib wrapper (implemented)
│   │   └── hnsw_index.cpp
│   ├── watcher/               # Filesystem monitoring
│   │   └── watcher.hpp        # Abstract file watcher interface (*)
│   ├── mcp/                   # MCP protocol and tools
│   │   ├── protocol.hpp       # JSON-RPC 2.0 message types
│   │   ├── mcp_server.hpp     # MCP server (implemented)
│   │   ├── mcp_server.cpp
│   │   ├── tools.hpp          # Tool definitions (stub handlers)
│   │   └── tools.cpp
│   ├── session/               # Session memory management
│   │   ├── session_store.hpp  # Session storage (implemented)
│   │   ├── session_store.cpp
│   │   └── session_embedder.hpp  # Abstract session embedder interface (*)
│   └── main.cpp               # Entry point, CLI arg parsing, spdlog setup
├── tests/
│   ├── test_placeholder.cpp   # Build sanity checks
│   ├── test_chunker.cpp       # Regex chunker tests (19 cases)
│   ├── test_index.cpp         # HNSW index tests (12 cases)
│   └── test_mcp_protocol.cpp  # MCP server tests (24 cases)
└── data/                      # Persistent index data (gitignored)
    └── .gitkeep
```

### Build Targets

| CMake Target | Type | Sources |
|--------------|------|---------|
| `engram-mcp` | Executable | `main.cpp` |
| `engram_chunker` | Static lib | `regex_chunker.cpp` |
| `engram_session` | Static lib | `session_store.cpp` |
| `engram_index` | Static lib | `hnsw_index.cpp` |
| `engram_mcp_lib` | Static lib | `mcp_server.cpp`, `tools.cpp` |
| `engram_core` | Interface lib | Aggregates nlohmann/json, spdlog, hnswlib |
| `engram_tests` | Test exe | All `tests/*.cpp` |

### Not Yet Implemented (Planned)

- `src/embedder/ort_embedder.cpp/.hpp` — ONNX Runtime + CUDA EP inference
- `src/chunker/treesitter_chunker.cpp/.hpp` — Tree-sitter language-aware chunker
- `src/watcher/win_watcher.cpp/.hpp` — ReadDirectoryChangesW file watcher
- `src/session/session_embedder.cpp` — Session embedding implementation
- `tests/test_embedder.cpp` — Embedder tests

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
./build/bin/engram-mcp.exe --project . --model models/nomic-embed-code.onnx

# Export embedding model (requires Python + torch + transformers)
python scripts/export_model.py --model nomic --output models/ --validate
```

## Things NOT to Do

- Don't write to stdout (reserved for MCP protocol)
- Don't use platform-specific APIs outside of `watcher/` module
- Don't heap-allocate in the search hot path
- Don't load the full ONNX model per query — keep session alive
- Don't store model files in git
- Don't use `std::map` for the chunk metadata store — use flat vectors or hash maps
