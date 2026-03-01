/// @file test_mcp_protocol.cpp
/// @brief Google Test suite for the MCP protocol layer.
///
/// Covers JSON-RPC parsing, response construction, tool registration,
/// message dispatch, and error handling.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "mcp/protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/tools.hpp"

using json = nlohmann::json;
using namespace engram::mcp;

// =========================================================================
// protocol.hpp — parse_message
// =========================================================================

TEST(Protocol, ParseValidJson) {
    auto j = parse_message(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})");
    ASSERT_FALSE(j.is_null());
    EXPECT_EQ(j["method"], "initialize");
    EXPECT_EQ(j["id"], 1);
}

TEST(Protocol, ParseMalformedJsonReturnsNull) {
    auto j = parse_message("{not valid json}");
    EXPECT_TRUE(j.is_null());
}

TEST(Protocol, ParseEmptyStringReturnsNull) {
    auto j = parse_message("");
    EXPECT_TRUE(j.is_null());
}

// =========================================================================
// protocol.hpp — classify_message
// =========================================================================

TEST(Protocol, ClassifyRequest) {
    json msg = {{"jsonrpc", "2.0"}, {"id", 1}, {"method", "tools/list"}};
    EXPECT_EQ(classify_message(msg), MessageKind::REQUEST);
}

TEST(Protocol, ClassifyNotification) {
    json msg = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
    EXPECT_EQ(classify_message(msg), MessageKind::NOTIFICATION);
}

TEST(Protocol, ClassifyResponse) {
    json msg = {{"jsonrpc", "2.0"}, {"id", 1}, {"result", json::object()}};
    EXPECT_EQ(classify_message(msg), MessageKind::RESPONSE);
}

TEST(Protocol, ClassifyInvalidNonObject) {
    EXPECT_EQ(classify_message(json(42)), MessageKind::INVALID);
    EXPECT_EQ(classify_message(json("string")), MessageKind::INVALID);
    EXPECT_EQ(classify_message(json::array()), MessageKind::INVALID);
}

TEST(Protocol, ClassifyInvalidEmptyObject) {
    EXPECT_EQ(classify_message(json::object()), MessageKind::INVALID);
}

// =========================================================================
// protocol.hpp — make_response
// =========================================================================

TEST(Protocol, MakeResponseStructure) {
    auto resp = make_response(42, {{"key", "value"}});

    EXPECT_EQ(resp["jsonrpc"], "2.0");
    EXPECT_EQ(resp["id"], 42);
    EXPECT_EQ(resp["result"]["key"], "value");
    EXPECT_FALSE(resp.contains("error"));
}

TEST(Protocol, MakeResponseWithStringId) {
    auto resp = make_response("abc-123", json::object());

    EXPECT_EQ(resp["id"], "abc-123");
    EXPECT_TRUE(resp["result"].is_object());
}

// =========================================================================
// protocol.hpp — make_error
// =========================================================================

TEST(Protocol, MakeErrorStructure) {
    auto err = make_error(7, ErrorCode::METHOD_NOT_FOUND, "no such method");

    EXPECT_EQ(err["jsonrpc"], "2.0");
    EXPECT_EQ(err["id"], 7);
    EXPECT_TRUE(err.contains("error"));
    EXPECT_EQ(err["error"]["code"], -32601);
    EXPECT_EQ(err["error"]["message"], "no such method");
    EXPECT_FALSE(err.contains("result"));
}

TEST(Protocol, MakeErrorWithNullId) {
    auto err = make_error(nullptr, ErrorCode::PARSE_ERROR, "parse failure");

    EXPECT_TRUE(err["id"].is_null());
    EXPECT_EQ(err["error"]["code"], -32700);
}

TEST(Protocol, MakeErrorWithRawIntCode) {
    auto err = make_error(1, -32603, "internal error");

    EXPECT_EQ(err["error"]["code"], -32603);
}

// =========================================================================
// ToolDefinition — serialization round-trip
// =========================================================================

TEST(Protocol, ToolDefinitionSerialization) {
    ToolDefinition td{
        "test_tool",
        "A test tool",
        {{"type", "object"}, {"properties", json::object()}}
    };

    json j = td;
    EXPECT_EQ(j["name"], "test_tool");
    EXPECT_EQ(j["description"], "A test tool");
    EXPECT_TRUE(j["inputSchema"].is_object());

    auto round_tripped = j.get<ToolDefinition>();
    EXPECT_EQ(round_tripped.name, td.name);
    EXPECT_EQ(round_tripped.description, td.description);
}

// =========================================================================
// McpServer — tool registration and listing
// =========================================================================

TEST(McpServer, RegisterToolAppearsInList) {
    McpServer server;

    server.register_tool(
        "my_tool",
        "does things",
        {{"type", "object"}},
        [](const json&) { return json{{"ok", true}}; }
    );

    ASSERT_EQ(server.tools().size(), 1u);
    EXPECT_EQ(server.tools()[0].definition.name, "my_tool");
    EXPECT_EQ(server.tools()[0].definition.description, "does things");
}

TEST(McpServer, ToolsListReturnsAllRegistered) {
    McpServer server;

    server.register_tool("a", "tool a", {{"type", "object"}},
                         [](const json&) { return json{}; });
    server.register_tool("b", "tool b", {{"type", "object"}},
                         [](const json&) { return json{}; });

    // Simulate a tools/list request.
    json request = {
        {"jsonrpc", "2.0"},
        {"id", 1},
        {"method", "tools/list"},
        {"params", json::object()}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ((*response)["id"], 1);

    auto tools = (*response)["result"]["tools"];
    ASSERT_EQ(tools.size(), 2u);
    EXPECT_EQ(tools[0]["name"], "a");
    EXPECT_EQ(tools[1]["name"], "b");
}

// =========================================================================
// McpServer — initialize
// =========================================================================

TEST(McpServer, InitializeReturnsServerInfo) {
    McpServer server;

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 1},
        {"method", "initialize"},
        {"params", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", json::object()},
            {"clientInfo", {{"name", "test"}, {"version", "0.0.0"}}}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());

    auto result = (*response)["result"];
    EXPECT_EQ(result["serverInfo"]["name"], "engram");
    EXPECT_EQ(result["serverInfo"]["version"], "0.1.0");
    EXPECT_TRUE(result["capabilities"].contains("tools"));
    EXPECT_TRUE(result.contains("protocolVersion"));
}

// =========================================================================
// McpServer — notifications/initialized returns no response
// =========================================================================

TEST(McpServer, InitializedNotificationNoResponse) {
    McpServer server;

    json notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"}
    };

    auto response = server.handle_message(notification);
    EXPECT_FALSE(response.has_value());
}

// =========================================================================
// McpServer — tools/call dispatches correctly
// =========================================================================

TEST(McpServer, ToolsCallDispatchesToHandler) {
    McpServer server;

    bool handler_called = false;
    json received_args;

    server.register_tool(
        "echo_tool",
        "echoes input",
        {{"type", "object"}, {"properties", {{"msg", {{"type", "string"}}}}}},
        [&](const json& args) -> json {
            handler_called = true;
            received_args = args;
            return {{"echoed", args.value("msg", "")}};
        }
    );

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 42},
        {"method", "tools/call"},
        {"params", {
            {"name", "echo_tool"},
            {"arguments", {{"msg", "hello"}}}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    EXPECT_TRUE(handler_called);
    EXPECT_EQ(received_args["msg"], "hello");

    // The response should have the content array wrapper.
    auto content = (*response)["result"]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 1u);
    EXPECT_EQ(content[0]["type"], "text");

    // The "text" field contains the JSON-serialized tool result.
    auto tool_result = json::parse(content[0]["text"].get<std::string>());
    EXPECT_EQ(tool_result["echoed"], "hello");
}

TEST(McpServer, ToolsCallUnknownToolReturnsError) {
    McpServer server;

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 5},
        {"method", "tools/call"},
        {"params", {
            {"name", "nonexistent_tool"},
            {"arguments", json::object()}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    EXPECT_TRUE((*response).contains("error"));
    EXPECT_EQ((*response)["error"]["code"], -32602);
}

TEST(McpServer, ToolsCallMissingNameReturnsError) {
    McpServer server;

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 6},
        {"method", "tools/call"},
        {"params", json::object()}  // no "name" field
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    EXPECT_TRUE((*response).contains("error"));
    EXPECT_EQ((*response)["error"]["code"], -32602);
}

// =========================================================================
// McpServer — unknown method returns METHOD_NOT_FOUND
// =========================================================================

TEST(McpServer, UnknownMethodReturnsError) {
    McpServer server;

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 99},
        {"method", "completely/unknown"},
        {"params", json::object()}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ((*response)["error"]["code"], -32601);
    EXPECT_EQ((*response)["id"], 99);
}

TEST(McpServer, UnknownNotificationSilentlyIgnored) {
    McpServer server;

    json notification = {
        {"jsonrpc", "2.0"},
        {"method", "unknown/notification"}
    };

    auto response = server.handle_message(notification);
    EXPECT_FALSE(response.has_value());
}

// =========================================================================
// McpServer — invalid messages
// =========================================================================

TEST(McpServer, InvalidMessageReturnsError) {
    McpServer server;

    // An empty object has neither "id" nor "method" — classified as INVALID.
    json bad_msg = json::object();

    auto response = server.handle_message(bad_msg);
    ASSERT_TRUE(response.has_value());
    EXPECT_TRUE((*response).contains("error"));
    EXPECT_EQ((*response)["error"]["code"], -32600);
}

// =========================================================================
// tools.cpp — register_all_tools with ToolContext
// =========================================================================

TEST(Tools, RegisterAllToolsPopulatesServer) {
    McpServer server;
    ToolContext ctx;  // All pointers nullptr — tools that need backends will return errors.
    register_all_tools(server, ctx);

    // We expect exactly 5 tools.
    ASSERT_EQ(server.tools().size(), 5u);

    // Collect names for checking.
    std::vector<std::string> names;
    for (const auto& t : server.tools()) {
        names.push_back(t.definition.name);
    }

    EXPECT_NE(std::find(names.begin(), names.end(), "search_code"),          names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "search_symbol"),        names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "get_context"),          names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "get_session_memory"),   names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "save_session_summary"), names.end());
}

TEST(Tools, SearchCodeWithoutEmbedderReturnsError) {
    McpServer server;
    ToolContext ctx;  // embedder is nullptr
    register_all_tools(server, ctx);

    // Call search_code via the server dispatch.
    json request = {
        {"jsonrpc", "2.0"},
        {"id", 1},
        {"method", "tools/call"},
        {"params", {
            {"name", "search_code"},
            {"arguments", {{"query", "test query"}}}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    ASSERT_FALSE((*response).contains("error"));

    auto content = (*response)["result"]["content"];
    ASSERT_TRUE(content.is_array());

    auto tool_result = json::parse(content[0]["text"].get<std::string>());
    EXPECT_TRUE(tool_result.contains("error"));
    EXPECT_TRUE(tool_result["error"].get<std::string>().find("embedder") != std::string::npos);
}

TEST(Tools, SearchSymbolWithoutChunkStoreReturnsError) {
    McpServer server;
    ToolContext ctx;  // chunk_store is nullptr
    register_all_tools(server, ctx);

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 2},
        {"method", "tools/call"},
        {"params", {
            {"name", "search_symbol"},
            {"arguments", {{"name", "MyClass"}}}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    ASSERT_FALSE((*response).contains("error"));

    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());
    EXPECT_TRUE(tool_result.contains("error"));
    EXPECT_TRUE(tool_result["error"].get<std::string>().find("chunk store") != std::string::npos);
}

TEST(Tools, SearchSymbolFindsMatchingChunks) {
    McpServer server;
    ToolContext ctx;

    // Populate a chunk store with test data.
    std::unordered_map<std::string, engram::Chunk> store;

    engram::Chunk c1;
    c1.chunk_id    = "chunk_001";
    c1.file_path   = "src/main.cpp";
    c1.start_line  = 10;
    c1.end_line    = 25;
    c1.language    = "cpp";
    c1.symbol_name = "MyClass";
    c1.source_text = "class MyClass {\npublic:\n    void doStuff();\n};";
    store["chunk_001"] = c1;

    engram::Chunk c2;
    c2.chunk_id    = "chunk_002";
    c2.file_path   = "src/utils.cpp";
    c2.start_line  = 5;
    c2.end_line    = 15;
    c2.language    = "cpp";
    c2.symbol_name = "helper_func";
    c2.source_text = "int helper_func(int x) { return x + 1; }";
    store["chunk_002"] = c2;

    ctx.chunk_store = &store;

    register_all_tools(server, ctx);

    // Search for "MyClass"
    json request = {
        {"jsonrpc", "2.0"},
        {"id", 3},
        {"method", "tools/call"},
        {"params", {
            {"name", "search_symbol"},
            {"arguments", {{"name", "MyClass"}}}
        }}
    };

    auto response = server.handle_message(request);
    ASSERT_TRUE(response.has_value());
    ASSERT_FALSE((*response).contains("error"));

    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());

    EXPECT_EQ(tool_result["count"], 1);
    ASSERT_EQ(tool_result["results"].size(), 1u);
    EXPECT_EQ(tool_result["results"][0]["chunk_id"], "chunk_001");
    EXPECT_EQ(tool_result["results"][0]["symbol_name"], "MyClass");
}

TEST(Tools, SearchSymbolCaseInsensitive) {
    McpServer server;
    ToolContext ctx;

    std::unordered_map<std::string, engram::Chunk> store;
    engram::Chunk c1;
    c1.chunk_id    = "chunk_001";
    c1.file_path   = "src/main.cpp";
    c1.start_line  = 1;
    c1.end_line    = 5;
    c1.language    = "cpp";
    c1.symbol_name = "MyFunction";
    c1.source_text = "void MyFunction() {}";
    store["chunk_001"] = c1;

    ctx.chunk_store = &store;
    register_all_tools(server, ctx);

    // Search with lowercase.
    json request = {
        {"jsonrpc", "2.0"},
        {"id", 4},
        {"method", "tools/call"},
        {"params", {
            {"name", "search_symbol"},
            {"arguments", {{"name", "myfunction"}}}
        }}
    };

    auto response = server.handle_message(request);
    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());

    EXPECT_EQ(tool_result["count"], 1);
}

TEST(Tools, GetContextFindsChunksInRange) {
    McpServer server;
    ToolContext ctx;

    std::unordered_map<std::string, engram::Chunk> store;

    engram::Chunk c1;
    c1.chunk_id    = "chunk_001";
    c1.file_path   = "src/main.cpp";
    c1.start_line  = 10;
    c1.end_line    = 20;
    c1.language    = "cpp";
    c1.symbol_name = "foo";
    c1.source_text = "void foo() {}";
    store["chunk_001"] = c1;

    engram::Chunk c2;
    c2.chunk_id    = "chunk_002";
    c2.file_path   = "src/main.cpp";
    c2.start_line  = 100;
    c2.end_line    = 120;
    c2.language    = "cpp";
    c2.symbol_name = "bar";
    c2.source_text = "void bar() {}";
    store["chunk_002"] = c2;

    ctx.chunk_store = &store;
    register_all_tools(server, ctx);

    // Query for line 15 with radius 10 -- should match c1 but not c2.
    json request = {
        {"jsonrpc", "2.0"},
        {"id", 5},
        {"method", "tools/call"},
        {"params", {
            {"name", "get_context"},
            {"arguments", {{"file", "src/main.cpp"}, {"line", 15}, {"radius", 10}}}
        }}
    };

    auto response = server.handle_message(request);
    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());

    EXPECT_EQ(tool_result["count"], 1);
    ASSERT_EQ(tool_result["results"].size(), 1u);
    EXPECT_EQ(tool_result["results"][0]["chunk_id"], "chunk_001");
}

TEST(Tools, SessionMemoryWithoutStoreReturnsError) {
    McpServer server;
    ToolContext ctx;  // session_store is nullptr
    register_all_tools(server, ctx);

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 6},
        {"method", "tools/call"},
        {"params", {
            {"name", "get_session_memory"},
            {"arguments", {{"query", "test"}}}
        }}
    };

    auto response = server.handle_message(request);
    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());
    EXPECT_TRUE(tool_result.contains("error"));
}

TEST(Tools, SaveSessionWithoutStoreReturnsError) {
    McpServer server;
    ToolContext ctx;  // session_store is nullptr
    register_all_tools(server, ctx);

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 7},
        {"method", "tools/call"},
        {"params", {
            {"name", "save_session_summary"},
            {"arguments", {{"summary", "We did things"}}}
        }}
    };

    auto response = server.handle_message(request);
    auto content = (*response)["result"]["content"];
    auto tool_result = json::parse(content[0]["text"].get<std::string>());
    EXPECT_TRUE(tool_result.contains("error"));
}

TEST(Tools, SaveAndRetrieveSession) {
    // Create a temp directory for session storage.
    auto temp_dir = std::filesystem::temp_directory_path() / "engram_test_sessions";
    std::filesystem::create_directories(temp_dir);

    // Clean up any leftover files.
    for (auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        std::filesystem::remove(entry.path());
    }

    engram::SessionStore session_store(temp_dir);

    McpServer server;
    ToolContext ctx;
    ctx.session_store = &session_store;
    register_all_tools(server, ctx);

    // Save a session.
    json save_request = {
        {"jsonrpc", "2.0"},
        {"id", 8},
        {"method", "tools/call"},
        {"params", {
            {"name", "save_session_summary"},
            {"arguments", {
                {"summary", "Implemented vector search refactoring"},
                {"key_files", json::array({"src/index/hnsw_index.cpp"})},
                {"key_decisions", json::array({"Use cosine similarity"})}
            }}
        }}
    };

    auto save_response = server.handle_message(save_request);
    auto save_content = (*save_response)["result"]["content"];
    auto save_result = json::parse(save_content[0]["text"].get<std::string>());

    EXPECT_EQ(save_result["status"], "saved");
    EXPECT_TRUE(save_result.contains("session_id"));
    EXPECT_TRUE(save_result.contains("timestamp"));

    // Now retrieve the session.
    json get_request = {
        {"jsonrpc", "2.0"},
        {"id", 9},
        {"method", "tools/call"},
        {"params", {
            {"name", "get_session_memory"},
            {"arguments", {{"query", "vector search"}}}
        }}
    };

    auto get_response = server.handle_message(get_request);
    auto get_content = (*get_response)["result"]["content"];
    auto get_result = json::parse(get_content[0]["text"].get<std::string>());

    EXPECT_GE(get_result["count"].get<int>(), 1);
    ASSERT_GE(get_result["sessions"].size(), 1u);
    EXPECT_TRUE(get_result["sessions"][0]["summary"].get<std::string>().find("vector search") !=
                std::string::npos);

    // Clean up.
    std::filesystem::remove_all(temp_dir);
}

TEST(Tools, EachToolDefinitionHasInputSchema) {
    McpServer server;
    ToolContext ctx;
    register_all_tools(server, ctx);

    for (const auto& t : server.tools()) {
        SCOPED_TRACE("tool: " + t.definition.name);
        EXPECT_TRUE(t.definition.input_schema.contains("type"));
        EXPECT_EQ(t.definition.input_schema["type"], "object");
        EXPECT_TRUE(t.definition.input_schema.contains("properties"));
    }
}
