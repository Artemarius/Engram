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
├── .claude/
│   └── skills/
│       └── engram-search/         # Claude Code skill (auto-triggers on semantic queries)
│           ├── SKILL.md           # Skill prompt: when to use engram vs built-in tools
│           └── references/
│               └── tool-guide.md  # Detailed parameter reference and decision flowchart
├── CMakeLists.txt             # Build system with FetchContent deps
├── CLAUDE.md
├── README.md
├── models/                    # ONNX model files (gitignored, downloaded separately)
│   └── .gitkeep
├── scripts/
│   ├── export_model.py        # Export embedding model to ONNX
│   ├── test_embeddings.py     # Validate ONNX output vs PyTorch
│   └── mcp_test_server.py     # Minimal Python MCP server for connection testing
├── src/
│   ├── chunker/               # Code splitting into semantic units
│   │   ├── chunker.hpp        # Abstract chunker interface + Chunk struct
│   │   ├── chunk_store.hpp    # Chunk metadata persistence (JSON serialization)
│   │   ├── chunk_store.cpp
│   │   ├── regex_chunker.hpp  # Regex-based chunker (implemented)
│   │   └── regex_chunker.cpp
│   ├── embedder/              # ONNX Runtime inference (requires ENGRAM_USE_ONNX)
│   │   ├── embedder.hpp       # Abstract embedder interface
│   │   ├── tokenizer.hpp      # Abstract tokenizer interface
│   │   ├── ort_embedder.hpp   # ONNX Runtime embedder (implemented, pimpl)
│   │   ├── ort_embedder.cpp
│   │   ├── ort_tokenizer.hpp  # WordPiece tokenizer (implemented)
│   │   └── ort_tokenizer.cpp
│   ├── index/                 # Vector storage and search
│   │   ├── vector_index.hpp   # Abstract index interface
│   │   ├── hnsw_index.hpp     # hnswlib wrapper (implemented)
│   │   └── hnsw_index.cpp
│   ├── watcher/               # Filesystem monitoring
│   │   ├── watcher.hpp        # Abstract file watcher interface
│   │   ├── win_watcher.hpp    # Windows ReadDirectoryChangesW watcher (implemented)
│   │   └── win_watcher.cpp
│   ├── mcp/                   # MCP protocol and tools
│   │   ├── protocol.hpp       # JSON-RPC 2.0 message types
│   │   ├── mcp_server.hpp     # MCP server (implemented)
│   │   ├── mcp_server.cpp
│   │   ├── tools.hpp          # ToolContext + tool definitions (thread-safe)
│   │   └── tools.cpp
│   ├── session/               # Session memory management
│   │   ├── session_store.hpp  # Session storage (implemented)
│   │   ├── session_store.cpp
│   │   ├── session_embedder.hpp       # Abstract session embedder interface
│   │   ├── session_embedder_impl.hpp  # Concrete session embedder (HNSW-backed)
│   │   └── session_embedder_impl.cpp
│   └── main.cpp               # Entry point, CLI args, startup, watcher, MCP loop
├── tests/
│   ├── test_placeholder.cpp        # Build sanity checks (2 cases)
│   ├── test_chunker.cpp            # Regex chunker tests (23 cases)
│   ├── test_index.cpp              # HNSW index tests (12 cases)
│   ├── test_mcp_protocol.cpp       # MCP server + tool handler tests (34 cases)
│   ├── test_watcher.cpp            # File watcher tests (29 cases)
│   ├── test_embedder.cpp           # Tokenizer (20) + ORT embedder tests (5; 4 need model file)
│   └── test_session_embedder.cpp   # Session embedder tests (24 cases, mock embedder)
└── data/                      # Persistent index data (gitignored)
    └── .gitkeep
```

### Build Targets

| CMake Target | Type | Sources |
|--------------|------|---------|
| `engram-mcp` | Executable | `main.cpp` |
| `engram_chunker` | Static lib | `regex_chunker.cpp`, `chunk_store.cpp` |
| `engram_index` | Static lib | `hnsw_index.cpp` |
| `engram_session` | Static lib | `session_store.cpp`, `session_embedder_impl.cpp` (depends on `engram_index`) |
| `engram_watcher` | Static lib | `win_watcher.cpp` |
| `engram_mcp_lib` | Static lib | `mcp_server.cpp`, `tools.cpp` |
| `engram_embedder` | Static lib (conditional) | `ort_embedder.cpp`, `ort_tokenizer.cpp` (requires `ENGRAM_USE_ONNX`) |
| `engram_core` | Interface lib | Aggregates nlohmann/json, spdlog, hnswlib |
| `engram_tests` | Test exe | All `tests/*.cpp` (149 test cases total) |

### Not Yet Implemented (Planned)

- `src/chunker/treesitter_chunker.cpp/.hpp` — Tree-sitter language-aware chunker

## Key Technical Decisions

### Embedding Model
- Using `all-MiniLM-L6-v2` (384 dim, ~86MB ONNX) — lightweight, runs well on 6GB VRAM
- `nomic-ai/nomic-embed-code` was evaluated but is a 7B-param model (3584 dim, 27GB) — too large for 6GB GPU
- Export to ONNX via `scripts/export_model.py`, optionally INT8 quantized or FP16
- Run via ONNX Runtime C++ API with CUDA Execution Provider
- Keep model files out of git — export script + tokenizer saved alongside

### Vector Index
- hnswlib (header-only C++, no dependencies)
- Cosine similarity space
- Persist index to disk, reload on startup
- Incremental: add/remove embeddings as files change
- Parameters: M=16, efConstruction=200, efSearch=50 (tune later)

### Code Chunking Strategy
- Prefer tree-sitter for language-aware parsing (functions, classes, methods) — planned
- Regex-based splitting implemented for 9 languages (cpp, python, js, ts, java, rust, go, ruby, csharp)
- Blank-line splitting fallback for unknown languages
- Each chunk: source text, file path, line range, language, symbol name if available
- Target chunk size: 50-500 tokens (configurable)
- Tiny blocks merged into predecessors, but named blocks (functions/classes) are never merged into another named block (preserves symbol identity)
- Store chunk metadata alongside embedding in the index

### MCP Protocol
- Communicate over stdio (stdin/stdout) using JSON-RPC 2.0
- **MCP stdio transport uses newline-delimited JSON** (`{json}\n`), NOT Content-Length framing
  - Messages MUST NOT contain embedded newlines
  - On read, Content-Length framing is also accepted as a fallback
- On Windows, `WriteFile` + `FlushFileBuffers` is used for stdout to bypass C runtime buffering on pipes
- NEVER write anything to stdout except MCP protocol messages
- All logging goes to stderr via spdlog
- `tools/list` and `tools/call` handlers are implemented
- Tool responses return code snippets with file paths and line numbers
- Five tools implemented: `search_code`, `search_symbol`, `get_context`, `get_session_memory`, `save_session_summary`
- `ToolContext` struct injects backend components (embedder, index, session store, chunk store) into tool handlers
- Tool handlers are thread-safe: `OptionalLock` guards chunk_store reads against concurrent watcher writes via `ToolContext.shared_mutex`

### File Watcher Integration
- `WinFileWatcher` monitors the project directory after initial indexing
- Callback filters by supported extensions and skip directories
- Created/Modified/Renamed: re-chunks file, removes old chunks, inserts new ones (with optional embedding)
- Deleted: removes all chunks for that file
- Thread safety: `std::mutex` protects `chunk_map` and `vector_index`; chunking and embedding happen outside the lock to minimize contention

### Persistence
- HNSW vector index saved/loaded to `data_dir/index/`
- Chunk metadata saved/loaded to `data_dir/chunks.json` (atomic write via tmp+rename)
- On startup, if both index and chunks load successfully, re-indexing is skipped (warm restart)
- `--reindex` flag forces full re-index regardless of persisted state
- Both are saved on shutdown after watcher is stopped

### Session Memory
- On session end, accept a summary string from Claude Code
- `SessionEmbedderImpl` embeds session summaries into a dedicated HNSW index (separate from code chunks)
- Composed text combines summary + key_files + key_decisions for embedding
- On session start, semantic search retrieves relevant past session context
- Falls back to word-level keyword matching when embedder is unavailable (query is split into words; all words must appear somewhere in the combined session text)
- Store as JSON: { timestamp, summary, key_files, key_decisions }

### Claude Code Skill Integration
- `.claude/skills/engram-search/` ships with the repo (un-ignored in `.gitignore`)
- Skill auto-triggers on semantic/exploratory code questions
- Teaches Claude when to use engram MCP tools vs built-in Grep/Glob/Read
- `references/tool-guide.md` provides detailed parameter reference (progressive disclosure)
- Session memory workflow: retrieve at session start, save at session end

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
| hnswlib v0.8.0 | Vector index | FetchContent (GitHub) |
| nlohmann/json v3.11.3 | JSON parsing | FetchContent |
| spdlog v1.14.1 | Logging | FetchContent |
| ONNX Runtime 1.24.2 | ML inference (CUDA EP) | Pre-built GPU package (`ENGRAM_USE_ONNX`) |
| cuDNN 9.x | Required by ORT CUDA EP | Pre-built, DLLs co-located with ORT |
| tree-sitter | Code parsing (planned) | FetchContent (`ENGRAM_USE_TREESITTER`) |
| Google Test v1.14.0 | Testing | FetchContent |

## Build Commands

```bash
# Configure (without ONNX — core modules only)
cmake -B build -G "Visual Studio 17 2022" -A x64

# Configure (with ONNX Runtime for GPU-accelerated embedding)
cmake -B build -G "Visual Studio 17 2022" -A x64 \
  -DENGRAM_USE_ONNX=ON \
  -DONNXRUNTIME_ROOT="D:/SDKs/onnxruntime-win-x64-gpu-1.24.2"

# Build (post-build step auto-copies ORT + cuDNN DLLs to bin/)
cmake --build build --config Release

# Run tests
cd build && ctest -C Release --output-on-failure

# Run the MCP server (for testing)
./build/bin/engram-mcp.exe --project . --model models/all-MiniLM-L6-v2.onnx

# Export embedding model (requires Python + torch + transformers)
# Set HF_HOME=D:\HFCache first to avoid downloading models to C:
python scripts/export_model.py --model minilm --output models/ --validate
```

## Things NOT to Do

- Don't write to stdout (reserved for MCP protocol)
- Don't use platform-specific APIs outside of `watcher/` and `mcp/` modules (mcp_server.cpp uses Win32 WriteFile for pipe I/O)
- Don't heap-allocate in the search hot path
- Don't load the full ONNX model per query — keep session alive
- Don't store model files in git
- Don't use `std::map` for the chunk metadata store — use flat vectors or hash maps
