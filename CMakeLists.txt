cmake_minimum_required(VERSION 3.24)
project(engram LANGUAGES CXX C)   # C required for tree-sitter

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Output directories — place binaries in bin/ and libs in lib/
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Multi-config generators (Visual Studio) use per-config subdirs by default.
# Force all configs into the same flat folder so the path is predictable.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${config} config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
endforeach()

# ---------------------------------------------------------------------------
# MSVC-specific compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /utf-8)
    # Silence C4100 (unreferenced formal parameter) — common during early dev.
    add_compile_options(/wd4100)
endif()

# ---------------------------------------------------------------------------
# Optional heavyweight dependencies — OFF by default so the core always builds
# ---------------------------------------------------------------------------
option(ENGRAM_USE_ONNX       "Enable ONNX Runtime (CUDA EP) for embedding inference" OFF)
option(ENGRAM_USE_TREESITTER "Enable tree-sitter for language-aware code chunking"    OFF)

# ---------------------------------------------------------------------------
# FetchContent — lightweight dependencies we always pull
# ---------------------------------------------------------------------------
include(FetchContent)

# nlohmann/json  (v3.11.3)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

# spdlog  (v1.14.1)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
)

# hnswlib — header-only vector index
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG        v0.8.0
    GIT_SHALLOW    TRUE
)

# Google Test  (v1.14.0)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)

# Prevent GTest from overriding the parent project's compiler/linker settings
# on Windows (avoids the /MTd vs /MDd mismatch issue).
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json spdlog hnswlib googletest)

# ---------------------------------------------------------------------------
# hnswlib is header-only and doesn't define a CMake target we can link.
# Create an INTERFACE library so downstream targets can just link it.
# ---------------------------------------------------------------------------
if(NOT TARGET hnswlib::hnswlib)
    add_library(hnswlib_iface INTERFACE)
    add_library(hnswlib::hnswlib ALIAS hnswlib_iface)
    target_include_directories(hnswlib_iface INTERFACE "${hnswlib_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime (optional)
# ---------------------------------------------------------------------------
if(ENGRAM_USE_ONNX)
    # Expects ONNXRUNTIME_ROOT to point at the pre-built GPU package, e.g.:
    #   cmake -DONNXRUNTIME_ROOT="D:/SDKs/onnxruntime-win-x64-gpu-1.24.2" ..
    #
    # The package layout (v1.24+) puts all files in lib/:
    #   <root>/include/          — public headers
    #   <root>/lib/              — onnxruntime.lib + onnxruntime.dll + provider DLLs
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        message(FATAL_ERROR
            "ENGRAM_USE_ONNX is ON but ONNXRUNTIME_ROOT is not set. "
            "Point it at the extracted ONNX Runtime GPU package directory.")
    endif()

    # Detect DLL location: older packages use bin/, newer (1.24+) use lib/.
    if(EXISTS "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll")
        set(_ORT_DLL_DIR "${ONNXRUNTIME_ROOT}/bin")
    else()
        set(_ORT_DLL_DIR "${ONNXRUNTIME_ROOT}/lib")
    endif()

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB   "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${_ORT_DLL_DIR}/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
    message(STATUS "ONNX Runtime enabled from: ${ONNXRUNTIME_ROOT}")
    message(STATUS "ONNX Runtime DLLs in: ${_ORT_DLL_DIR}")
endif()

# ---------------------------------------------------------------------------
# tree-sitter (optional) — AST-aware code chunking
# ---------------------------------------------------------------------------
if(ENGRAM_USE_TREESITTER)
    # We use FetchContent_Populate (not MakeAvailable) to download grammar
    # source without running their CMakeLists.txt — avoids target conflicts.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()

    # --- Core library (built from amalgamated source to avoid WASM issues) ---
    FetchContent_Declare(
        tree_sitter
        GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter.git
        GIT_TAG        v0.24.7
        GIT_SHALLOW    TRUE
    )
    # Populate only (don't run tree-sitter's own CMakeLists.txt).
    FetchContent_GetProperties(tree_sitter)
    if(NOT tree_sitter_POPULATED)
        FetchContent_Populate(tree_sitter)
    endif()

    add_library(tree_sitter_core STATIC
        "${tree_sitter_SOURCE_DIR}/lib/src/lib.c"
    )
    target_include_directories(tree_sitter_core
        PUBLIC  "${tree_sitter_SOURCE_DIR}/lib/include"
        PRIVATE "${tree_sitter_SOURCE_DIR}/lib/src"
    )
    set_target_properties(tree_sitter_core PROPERTIES C_STANDARD 11)
    if(MSVC)
        target_compile_options(tree_sitter_core PRIVATE /W0)
    endif()

    # --- Helper function: add a tree-sitter grammar library ---
    function(engram_add_grammar NAME REPO TAG HAS_SCANNER)
        FetchContent_Declare(
            tree_sitter_${NAME}
            GIT_REPOSITORY ${REPO}
            GIT_TAG        ${TAG}
            GIT_SHALLOW    TRUE
        )
        # Populate only — bypass their CMakeLists.txt (avoids target conflicts).
        FetchContent_GetProperties(tree_sitter_${NAME})
        if(NOT tree_sitter_${NAME}_POPULATED)
            FetchContent_Populate(tree_sitter_${NAME})
        endif()

        set(_GRAMMAR_SRC "${tree_sitter_${NAME}_SOURCE_DIR}/src/parser.c")
        if(HAS_SCANNER)
            if(EXISTS "${tree_sitter_${NAME}_SOURCE_DIR}/src/scanner.c")
                list(APPEND _GRAMMAR_SRC "${tree_sitter_${NAME}_SOURCE_DIR}/src/scanner.c")
            elseif(EXISTS "${tree_sitter_${NAME}_SOURCE_DIR}/src/scanner.cc")
                list(APPEND _GRAMMAR_SRC "${tree_sitter_${NAME}_SOURCE_DIR}/src/scanner.cc")
            endif()
        endif()

        add_library(tree_sitter_${NAME} STATIC ${_GRAMMAR_SRC})
        target_include_directories(tree_sitter_${NAME} PRIVATE
            "${tree_sitter_SOURCE_DIR}/lib/include"
            "${tree_sitter_${NAME}_SOURCE_DIR}/src"
        )
        set_target_properties(tree_sitter_${NAME} PROPERTIES C_STANDARD 11)
        if(MSVC)
            target_compile_options(tree_sitter_${NAME} PRIVATE /W0)
        endif()
    endfunction()

    # --- Language grammars (9 languages) ---
    engram_add_grammar(cpp
        "https://github.com/tree-sitter/tree-sitter-cpp.git"       v0.23.4  TRUE)
    engram_add_grammar(python
        "https://github.com/tree-sitter/tree-sitter-python.git"    v0.23.6  TRUE)
    engram_add_grammar(javascript
        "https://github.com/tree-sitter/tree-sitter-javascript.git" v0.23.1 TRUE)
    engram_add_grammar(java
        "https://github.com/tree-sitter/tree-sitter-java.git"      v0.23.5  FALSE)
    engram_add_grammar(rust
        "https://github.com/tree-sitter/tree-sitter-rust.git"      v0.23.3  TRUE)
    engram_add_grammar(go
        "https://github.com/tree-sitter/tree-sitter-go.git"        v0.23.4  FALSE)
    engram_add_grammar(ruby
        "https://github.com/tree-sitter/tree-sitter-ruby.git"      v0.23.1  TRUE)
    engram_add_grammar(c_sharp
        "https://github.com/tree-sitter/tree-sitter-c-sharp.git"   v0.23.1  TRUE)

    # --- TypeScript: two sub-parsers from one repo ---
    FetchContent_Declare(
        tree_sitter_typescript_repo
        GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-typescript.git
        GIT_TAG        v0.23.2
        GIT_SHALLOW    TRUE
    )
    # Populate only — bypass their CMakeLists.txt.
    FetchContent_GetProperties(tree_sitter_typescript_repo)
    if(NOT tree_sitter_typescript_repo_POPULATED)
        FetchContent_Populate(tree_sitter_typescript_repo)
    endif()

    # TypeScript sub-parser
    set(_TS_REPO "${tree_sitter_typescript_repo_SOURCE_DIR}")
    add_library(tree_sitter_typescript STATIC
        "${_TS_REPO}/typescript/src/parser.c"
        "${_TS_REPO}/typescript/src/scanner.c"
    )
    target_include_directories(tree_sitter_typescript PRIVATE
        "${tree_sitter_SOURCE_DIR}/lib/include"
        "${_TS_REPO}/typescript/src"
    )
    set_target_properties(tree_sitter_typescript PROPERTIES C_STANDARD 11)
    if(MSVC)
        target_compile_options(tree_sitter_typescript PRIVATE /W0)
    endif()

    # TSX sub-parser
    add_library(tree_sitter_tsx STATIC
        "${_TS_REPO}/tsx/src/parser.c"
        "${_TS_REPO}/tsx/src/scanner.c"
    )
    target_include_directories(tree_sitter_tsx PRIVATE
        "${tree_sitter_SOURCE_DIR}/lib/include"
        "${_TS_REPO}/tsx/src"
    )
    set_target_properties(tree_sitter_tsx PROPERTIES C_STANDARD 11)
    if(MSVC)
        target_compile_options(tree_sitter_tsx PRIVATE /W0)
    endif()

    # Collect all grammar libs into a single variable for linking.
    set(ENGRAM_TS_GRAMMAR_LIBS
        tree_sitter_cpp
        tree_sitter_python
        tree_sitter_javascript
        tree_sitter_typescript
        tree_sitter_tsx
        tree_sitter_java
        tree_sitter_rust
        tree_sitter_go
        tree_sitter_ruby
        tree_sitter_c_sharp
    )

    message(STATUS "tree-sitter enabled (core + ${ENGRAM_TS_GRAMMAR_LIBS})")
endif()

# ---------------------------------------------------------------------------
# engram_core — INTERFACE library that aggregates common dependencies.
# Module source files will be added to their own STATIC/OBJECT libraries
# later; this target gives them a single place to inherit include paths and
# link requirements.
# ---------------------------------------------------------------------------
add_library(engram_core INTERFACE)
target_include_directories(engram_core INTERFACE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(engram_core
    INTERFACE
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        hnswlib::hnswlib
)

if(ENGRAM_USE_ONNX)
    target_link_libraries(engram_core INTERFACE onnxruntime)
    target_compile_definitions(engram_core INTERFACE ENGRAM_HAS_ONNX=1)
endif()

if(ENGRAM_USE_TREESITTER)
    target_compile_definitions(engram_core INTERFACE ENGRAM_HAS_TREESITTER=1)
endif()

# ---------------------------------------------------------------------------
# engram_chunker — Code chunking module (static library)
# ---------------------------------------------------------------------------
add_library(engram_chunker STATIC
    src/chunker/regex_chunker.cpp
    src/chunker/chunk_store.cpp
)
target_link_libraries(engram_chunker PUBLIC engram_core)

# ---------------------------------------------------------------------------
# engram_treesitter — Tree-sitter chunker module (static library, optional)
# ---------------------------------------------------------------------------
if(ENGRAM_USE_TREESITTER)
    add_library(engram_treesitter STATIC
        src/chunker/treesitter_chunker.cpp
    )
    target_link_libraries(engram_treesitter
        PUBLIC  engram_core engram_chunker
        PRIVATE tree_sitter_core ${ENGRAM_TS_GRAMMAR_LIBS}
    )
    target_include_directories(engram_treesitter PRIVATE
        "${tree_sitter_SOURCE_DIR}/lib/include"
    )
endif()

# ---------------------------------------------------------------------------
# engram_index — HNSW vector index module (static library)
# ---------------------------------------------------------------------------
add_library(engram_index STATIC
    src/index/hnsw_index.cpp
)
target_link_libraries(engram_index PUBLIC engram_core)

# ---------------------------------------------------------------------------
# engram_session — Session memory module (static library)
# Depends on engram_index because SessionEmbedderImpl uses HnswIndex.
# ---------------------------------------------------------------------------
add_library(engram_session STATIC
    src/session/session_store.cpp
    src/session/session_embedder_impl.cpp
)
target_link_libraries(engram_session PUBLIC engram_core engram_index)

# ---------------------------------------------------------------------------
# engram_embedder — ONNX Runtime embedder module (static library, optional)
# ---------------------------------------------------------------------------
if(ENGRAM_USE_ONNX)
    add_library(engram_embedder STATIC
        src/embedder/ort_embedder.cpp
        src/embedder/ort_tokenizer.cpp
    )
    target_link_libraries(engram_embedder PUBLIC engram_core)
endif()

# ---------------------------------------------------------------------------
# engram_watcher — Filesystem watcher module (static library)
# ---------------------------------------------------------------------------
add_library(engram_watcher STATIC
    src/watcher/win_watcher.cpp
)
target_link_libraries(engram_watcher PUBLIC engram_core)

# ---------------------------------------------------------------------------
# engram_mcp_lib — MCP server module (static library)
# ---------------------------------------------------------------------------
add_library(engram_mcp_lib STATIC
    src/mcp/mcp_server.cpp
    src/mcp/tools.cpp
)
target_link_libraries(engram_mcp_lib PUBLIC engram_core engram_index engram_session engram_chunker)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(engram-mcp
    src/main.cpp
)
target_link_libraries(engram-mcp PRIVATE engram_core engram_chunker engram_session engram_index engram_watcher engram_mcp_lib)
if(ENGRAM_USE_TREESITTER)
    target_link_libraries(engram-mcp PRIVATE engram_treesitter)
endif()
if(ENGRAM_USE_ONNX)
    target_link_libraries(engram-mcp PRIVATE engram_embedder)

    # Copy ONNX Runtime (and co-located cuDNN) DLLs next to executables so they
    # are found at runtime without modifying PATH.
    file(GLOB _ORT_DLLS "${_ORT_DLL_DIR}/*.dll")
    foreach(_dll ${_ORT_DLLS})
        add_custom_command(TARGET engram-mcp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:engram-mcp>"
            COMMENT "Copying ${_dll}"
        )
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

set(ENGRAM_TEST_SOURCES
    tests/test_placeholder.cpp
    tests/test_chunker.cpp
    tests/test_index.cpp
    tests/test_mcp_protocol.cpp
    tests/test_watcher.cpp
    tests/test_embedder.cpp
    tests/test_session_embedder.cpp
)
if(ENGRAM_USE_TREESITTER)
    list(APPEND ENGRAM_TEST_SOURCES tests/test_treesitter_chunker.cpp)
endif()

add_executable(engram_tests ${ENGRAM_TEST_SOURCES})
target_link_libraries(engram_tests
    PRIVATE
        engram_core
        engram_chunker
        engram_session
        engram_index
        engram_watcher
        engram_mcp_lib
        GTest::gtest
        GTest::gtest_main
)
if(ENGRAM_USE_TREESITTER)
    target_link_libraries(engram_tests PRIVATE engram_treesitter)
endif()
if(ENGRAM_USE_ONNX)
    target_link_libraries(engram_tests PRIVATE engram_embedder)
endif()

include(GoogleTest)
gtest_discover_tests(engram_tests)

# ---------------------------------------------------------------------------
# Benchmarks (standalone executable, not part of CTest)
# ---------------------------------------------------------------------------
add_executable(engram_benchmarks
    benchmarks/bench_chunker.cpp
)
target_link_libraries(engram_benchmarks PRIVATE engram_core engram_chunker)
if(ENGRAM_USE_TREESITTER)
    target_link_libraries(engram_benchmarks PRIVATE engram_treesitter)
endif()
if(ENGRAM_USE_ONNX)
    target_link_libraries(engram_benchmarks PRIVATE engram_embedder engram_index)
endif()
