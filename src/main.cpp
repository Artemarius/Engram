#include <cstdlib>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stderr_color_sink.h>
#include <nlohmann/json.hpp>

/// Parse a simple --key value pair from the argument list.
/// Returns the value if found, or `fallback` otherwise.
static std::string parse_arg(const std::vector<std::string>& args,
                             const std::string& key,
                             const std::string& fallback = {})
{
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return fallback;
}

/// Check if a flag (no value) is present.
static bool has_flag(const std::vector<std::string>& args,
                     const std::string& flag)
{
    for (const auto& a : args) {
        if (a == flag) return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    // Logger — everything goes to stderr; stdout is reserved for MCP protocol.
    // -----------------------------------------------------------------------
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("engram", stderr_sink);
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);

    // -----------------------------------------------------------------------
    // Command-line arguments
    // -----------------------------------------------------------------------
    std::vector<std::string> args(argv, argv + argc);

    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        // Help text goes to stderr so stdout stays clean for MCP.
        spdlog::info("Usage: engram-mcp [options]");
        spdlog::info("  --project <path>   Root of the codebase to index");
        spdlog::info("  --model   <path>   Path to the ONNX embedding model");
        spdlog::info("  --verbose          Enable debug-level logging");
        return 0;
    }

    if (has_flag(args, "--verbose")) {
        logger->set_level(spdlog::level::debug);
    }

    const std::string project_path = parse_arg(args, "--project");
    const std::string model_path   = parse_arg(args, "--model");

    // -----------------------------------------------------------------------
    // Startup banner (stderr only)
    // -----------------------------------------------------------------------
    spdlog::info("engram-mcp starting up");
    spdlog::info("  build: {} {}", __DATE__, __TIME__);

    if (!project_path.empty()) {
        spdlog::info("  project: {}", project_path);
    } else {
        spdlog::warn("  no --project specified; indexing disabled");
    }

    if (!model_path.empty()) {
        spdlog::info("  model: {}", model_path);
    } else {
        spdlog::warn("  no --model specified; embedding disabled");
    }

    // Quick sanity check that nlohmann/json links correctly.
    nlohmann::json meta = {
        {"name",    "engram-mcp"},
        {"version", "0.1.0"}
    };
    spdlog::debug("server metadata: {}", meta.dump());

    // -----------------------------------------------------------------------
    // TODO: Wire up the MCP server (JSON-RPC over stdio) here.
    //       For now we just exit cleanly.
    // -----------------------------------------------------------------------
    spdlog::info("no MCP server loop implemented yet — exiting");

    return 0;
}
