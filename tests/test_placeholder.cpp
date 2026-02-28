#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

// Placeholder test to verify the build system and dependency linkage.
// Replace with real module tests as source files are added.

TEST(SanityCheck, JsonRoundTrip)
{
    nlohmann::json j = {
        {"key", "value"},
        {"number", 42}
    };

    std::string serialized = j.dump();
    auto parsed = nlohmann::json::parse(serialized);

    EXPECT_EQ(parsed["key"], "value");
    EXPECT_EQ(parsed["number"], 42);
}

TEST(SanityCheck, TrueIsTrue)
{
    EXPECT_TRUE(true);
}
