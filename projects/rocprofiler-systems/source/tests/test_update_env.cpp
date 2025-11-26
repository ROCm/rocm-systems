#include "common/environment.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace rocprofsys::common;

// Helper function to find environment variable in vector
std::string
find_env_var(const std::vector<char*>& env, std::string_view var_name)
{
    std::string prefix = std::string(var_name) + "=";
    for(auto* itr : env)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(prefix) == 0)
        {
            return std::string{ itr };
        }
    }
    return "";
}

class UpdateEnvTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear environment
        env_vars.clear();
        updated_envs.clear();
        original_envs.clear();
    }

    void TearDown() override
    {
        // Clean up allocated strings
        for(auto* ptr : env_vars)
        {
            if(ptr) free(ptr);
        }
    }

    std::vector<char*>                   env_vars;
    std::unordered_set<std::string_view> updated_envs;
    std::unordered_set<std::string>      original_envs;
};

TEST_F(UpdateEnvTest, ReplaceMode_NewVariable)
{
    update_env(env_vars, "TEST_VAR", "test_value", UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "TEST_VAR=test_value");
    EXPECT_EQ(updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, ReplaceMode_ExistingVariable)
{
    // Add initial variable
    env_vars.push_back(strdup("TEST_VAR=old_value"));
    original_envs.insert("TEST_VAR=old_value");

    // Replace it
    update_env(env_vars, "TEST_VAR", "new_value", UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "TEST_VAR=new_value");
    EXPECT_EQ(updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, AppendMode_NewVariable)
{
    update_env(env_vars, "PATH", "/new/path", UPD_APPEND, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "PATH=/new/path");
}

TEST_F(UpdateEnvTest, AppendMode_ExistingVariable)
{
    env_vars.push_back(strdup("PATH=/old/path"));
    original_envs.insert("PATH=/old/path");

    update_env(env_vars, "PATH", "/new/path", UPD_APPEND, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "PATH=/old/path:/new/path");
    EXPECT_EQ(updated_envs.count("PATH"), 1);
}

TEST_F(UpdateEnvTest, PrependMode_ExistingVariable)
{
    env_vars.push_back(strdup("LD_LIBRARY_PATH=/old/lib"));
    original_envs.insert("LD_LIBRARY_PATH=/old/lib");

    update_env(env_vars, "LD_LIBRARY_PATH", "/new/lib", UPD_PREPEND, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "LD_LIBRARY_PATH=/new/lib:/old/lib");
}

TEST_F(UpdateEnvTest, WeakMode_OriginalValue)
{
    env_vars.push_back(strdup("WEAK_VAR=original"));
    original_envs.insert("WEAK_VAR=original");

    // Weak update should succeed (value unchanged from original)
    update_env(env_vars, "WEAK_VAR", "new_value", UPD_WEAK, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "WEAK_VAR=new_value");
}

TEST_F(UpdateEnvTest, WeakMode_ModifiedValue)
{
    env_vars.push_back(strdup("WEAK_VAR=original"));
    original_envs.insert("WEAK_VAR=original");

    // Change the value first
    free(env_vars[0]);
    env_vars[0] = strdup("WEAK_VAR=modified");

    // Weak update should NOT succeed (value changed from original)
    update_env(env_vars, "WEAK_VAR", "new_value", UPD_WEAK, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "WEAK_VAR=modified");  // Should remain unchanged
}

TEST_F(UpdateEnvTest, BooleanValue_True)
{
    update_env(env_vars, "BOOL_VAR", true, UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "BOOL_VAR=true");
}

TEST_F(UpdateEnvTest, BooleanValue_False)
{
    update_env(env_vars, "BOOL_VAR", false, UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "BOOL_VAR=false");
}

TEST_F(UpdateEnvTest, NumericValue)
{
    update_env(env_vars, "NUM_VAR", 42, UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "NUM_VAR=42");
}

TEST_F(UpdateEnvTest, AppendMode_AvoidsDuplicates)
{
    env_vars.push_back(strdup("PATH=/existing/path"));
    original_envs.insert("PATH=/existing/path");

    // Try to append the same path
    update_env(env_vars, "PATH", "/existing/path", UPD_APPEND, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "PATH=/existing/path");  // Should not duplicate
}

TEST_F(UpdateEnvTest, PrependAndAppend_PrefersAppend)
{
    env_vars.push_back(strdup("VAR=/middle"));
    original_envs.insert("VAR=/middle");

    // Both flags set - should prefer append
    update_env(env_vars, "VAR", "/new",
               static_cast<update_mode>(UPD_PREPEND | UPD_APPEND), ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "VAR=/middle:/new");  // Appended, not prepended
}

TEST_F(UpdateEnvTest, CustomDelimiter)
{
    env_vars.push_back(strdup("VAR=a"));
    original_envs.insert("VAR=a");

    update_env(env_vars, "VAR", "b", UPD_APPEND, ",", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "VAR=a,b");
}

// Additional comprehensive tests for real-world scenarios

TEST_F(UpdateEnvTest, RealWorld_LD_LIBRARY_PATH_Append)
{
    env_vars.push_back(strdup("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib"));
    original_envs.insert("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib");

    update_env(env_vars, "LD_LIBRARY_PATH", "/opt/rocm/lib", UPD_APPEND, ":",
               updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/opt/rocm/lib");
}

TEST_F(UpdateEnvTest, RealWorld_LD_PRELOAD_Prepend)
{
    env_vars.push_back(strdup("LD_PRELOAD=/lib/existing.so"));
    original_envs.insert("LD_PRELOAD=/lib/existing.so");

    update_env(env_vars, "LD_PRELOAD", "/opt/rocm/librocprof-sys-dl.so", UPD_PREPEND, ":",
               updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0],
                 "LD_PRELOAD=/opt/rocm/librocprof-sys-dl.so:/lib/existing.so");
}

TEST_F(UpdateEnvTest, RealWorld_ROCPROFSYS_Environment_Variables)
{
    // Test boolean environment variables commonly used in rocprof-sys
    update_env(env_vars, "ROCPROFSYS_TRACE", true, UPD_REPLACE, ":", updated_envs,
               original_envs);
    update_env(env_vars, "ROCPROFSYS_PROFILE", false, UPD_REPLACE, ":", updated_envs,
               original_envs);
    update_env(env_vars, "ROCPROFSYS_USE_SAMPLING", true, UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 3);
    EXPECT_STREQ(find_env_var(env_vars, "ROCPROFSYS_TRACE").c_str(),
                 "ROCPROFSYS_TRACE=true");
    EXPECT_STREQ(find_env_var(env_vars, "ROCPROFSYS_PROFILE").c_str(),
                 "ROCPROFSYS_PROFILE=false");
    EXPECT_STREQ(find_env_var(env_vars, "ROCPROFSYS_USE_SAMPLING").c_str(),
                 "ROCPROFSYS_USE_SAMPLING=true");
}

TEST_F(UpdateEnvTest, RealWorld_Timing_DoubleValues)
{
    // Test double values used for timing parameters
    update_env(env_vars, "ROCPROFSYS_TRACE_DELAY", 1.5, UPD_REPLACE, ":", updated_envs,
               original_envs);
    update_env(env_vars, "ROCPROFSYS_SAMPLING_FREQ", 100.0, UPD_REPLACE, ":",
               updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 2);
    std::string delay_var = find_env_var(env_vars, "ROCPROFSYS_TRACE_DELAY");
    std::string freq_var  = find_env_var(env_vars, "ROCPROFSYS_SAMPLING_FREQ");

    // Check that the variables exist and contain numeric values
    EXPECT_TRUE(delay_var.find("ROCPROFSYS_TRACE_DELAY=") == 0);
    EXPECT_TRUE(freq_var.find("ROCPROFSYS_SAMPLING_FREQ=") == 0);
}

TEST_F(UpdateEnvTest, StringTypes_StdString)
{
    std::string value = "test_string_value";
    update_env(env_vars, "STRING_VAR", value, UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "STRING_VAR=test_string_value");
}

TEST_F(UpdateEnvTest, StringTypes_ConstCharPtr)
{
    const char* value = "const_char_value";
    update_env(env_vars, "CHAR_VAR", value, UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "CHAR_VAR=const_char_value");
}

TEST_F(UpdateEnvTest, EmptyStringValue)
{
    update_env(env_vars, "EMPTY_VAR", "", UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "EMPTY_VAR=");
}

TEST_F(UpdateEnvTest, MultipleVariables_DifferentNames)
{
    update_env(env_vars, "VAR1", "value1", UPD_REPLACE, ":", updated_envs, original_envs);
    update_env(env_vars, "VAR2", "value2", UPD_REPLACE, ":", updated_envs, original_envs);
    update_env(env_vars, "VAR3", "value3", UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 3);
    EXPECT_EQ(updated_envs.size(), 3);
    EXPECT_STREQ(find_env_var(env_vars, "VAR1").c_str(), "VAR1=value1");
    EXPECT_STREQ(find_env_var(env_vars, "VAR2").c_str(), "VAR2=value2");
    EXPECT_STREQ(find_env_var(env_vars, "VAR3").c_str(), "VAR3=value3");
}

TEST_F(UpdateEnvTest, NullPointer_InEnvironmentVector)
{
    // Add some variables with nullptr in between
    env_vars.push_back(strdup("VAR1=value1"));
    env_vars.push_back(nullptr);  // Null pointer
    env_vars.push_back(strdup("VAR2=value2"));
    original_envs.insert("VAR1=value1");
    original_envs.insert("VAR2=value2");

    // Update existing variable (should skip nullptr)
    update_env(env_vars, "VAR2", "new_value2", UPD_REPLACE, ":", updated_envs,
               original_envs);

    EXPECT_STREQ(find_env_var(env_vars, "VAR2").c_str(), "VAR2=new_value2");
}

TEST_F(UpdateEnvTest, LongPath_Append)
{
    std::string long_path = "/very/long/path/to/some/directory/with/many/subdirectories/"
                            "that/might/be/used/in/real/world";
    env_vars.push_back(strdup("PATH=/usr/bin:/bin"));
    original_envs.insert("PATH=/usr/bin:/bin");

    update_env(env_vars, "PATH", long_path, UPD_APPEND, ":", updated_envs, original_envs);

    std::string expected = "PATH=/usr/bin:/bin:" + long_path;
    EXPECT_STREQ(env_vars[0], expected.c_str());
}

TEST_F(UpdateEnvTest, SpecialCharacters_InValue)
{
    // Test values with special characters
    update_env(env_vars, "SPECIAL_VAR", "value-with_special.chars:123", UPD_REPLACE, ":",
               updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "SPECIAL_VAR=value-with_special.chars:123");
}

TEST_F(UpdateEnvTest, IntegerValues_Positive)
{
    update_env(env_vars, "INT_VAR", 12345, UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "INT_VAR=12345");
}

TEST_F(UpdateEnvTest, IntegerValues_Negative)
{
    update_env(env_vars, "NEGATIVE_VAR", -999, UPD_REPLACE, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "NEGATIVE_VAR=-999");
}

TEST_F(UpdateEnvTest, IntegerValues_Zero)
{
    update_env(env_vars, "ZERO_VAR", 0, UPD_REPLACE, ":", updated_envs, original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "ZERO_VAR=0");
}

TEST_F(UpdateEnvTest, UpdateTracking_MultipleUpdates)
{
    // Test that updated_envs correctly tracks all updates
    update_env(env_vars, "VAR1", "val1", UPD_REPLACE, ":", updated_envs, original_envs);
    update_env(env_vars, "VAR2", "val2", UPD_REPLACE, ":", updated_envs, original_envs);
    update_env(env_vars, "VAR1", "val1_updated", UPD_REPLACE, ":", updated_envs,
               original_envs);  // Update VAR1 again

    EXPECT_EQ(updated_envs.count("VAR1"), 1);
    EXPECT_EQ(updated_envs.count("VAR2"), 1);
    EXPECT_EQ(updated_envs.size(), 2);  // Still only 2 unique variables
}

TEST_F(UpdateEnvTest, WeakMode_CombinedWithReplace)
{
    env_vars.push_back(strdup("CONFIG_VAR=initial"));
    original_envs.insert("CONFIG_VAR=initial");

    // First weak update should succeed
    update_env(env_vars, "CONFIG_VAR", "weak_update",
               static_cast<update_mode>(UPD_WEAK | UPD_REPLACE), ":", updated_envs,
               original_envs);

    EXPECT_STREQ(env_vars[0], "CONFIG_VAR=weak_update");

    // Change value
    free(env_vars[0]);
    env_vars[0] = strdup("CONFIG_VAR=user_modified");

    // Second weak update should NOT succeed (value changed)
    update_env(env_vars, "CONFIG_VAR", "another_weak_update",
               static_cast<update_mode>(UPD_WEAK | UPD_REPLACE), ":", updated_envs,
               original_envs);

    EXPECT_STREQ(env_vars[0], "CONFIG_VAR=user_modified");
}

TEST_F(UpdateEnvTest, Append_MultiplePathsInSequence)
{
    // Simulate building up a PATH variable
    update_env(env_vars, "BUILD_PATH", "/path1", UPD_REPLACE, ":", updated_envs,
               original_envs);
    update_env(env_vars, "BUILD_PATH", "/path2", UPD_APPEND, ":", updated_envs,
               original_envs);
    update_env(env_vars, "BUILD_PATH", "/path3", UPD_APPEND, ":", updated_envs,
               original_envs);
    update_env(env_vars, "BUILD_PATH", "/path4", UPD_APPEND, ":", updated_envs,
               original_envs);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "BUILD_PATH=/path1:/path2:/path3:/path4");
}