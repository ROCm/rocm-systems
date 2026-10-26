// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/cli.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <random>
#include <type_traits>

namespace util::cli {
namespace {

class CliTest : public testing::Test {
protected:
  Command root{"tool"};
  util::StringDiagnostic diagnostic;

  FailureOr<ParsedArguments> run(std::initializer_list<std::string_view> args) {
    return parse(root, {args.begin(), args.size()}, diagnostic.emitter());
  }
};

TEST_F(CliTest, TypedAndStringUpdatesShareOneValueAndAliases) {
  auto count = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c", "--limit"}, "Count"}, 64);
  ASSERT_TRUE(root.add_flag(count, diagnostic.emitter()).succeeded());
  EXPECT_EQ(root.find_flag("-c"), root.find_flag("--count"));
  EXPECT_EQ(root.flags().size(), 1u);
  EXPECT_EQ(root.flags()[0]->info().aliases.size(), 2u);
  ASSERT_TRUE(run({"-c", "128", "input"}).succeeded());
  EXPECT_EQ(count->value(), 128);
  ASSERT_TRUE(count->set_value(-42).succeeded());
  EXPECT_EQ(root.find_flag("--limit")->format_value(), "-42");
  ASSERT_TRUE(root.find_flag("--limit")->parse_and_set("256", diagnostic.emitter()).succeeded());
  EXPECT_EQ(count->value(), 256);
  EXPECT_TRUE(root.find_flag("--limit")->parse_and_set("bad", diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("--count"), std::string::npos);
}

TEST_F(CliTest, NestedScopesPositionalsAndSeparators) {
  auto global = std::make_shared<BoolFlag>(FlagInfo{"--verbose", {}, "Verbose", ValueMode::Switch});
  ASSERT_TRUE(root.add_flag(global, diagnostic.emitter()).succeeded());
  auto inspect = root.add_command("inspect", diagnostic.emitter());
  ASSERT_TRUE(inspect.succeeded());
  auto code = inspect.value()->add_command("code", diagnostic.emitter());
  ASSERT_TRUE(code.succeeded());
  auto limit = std::make_shared<IntFlag>(FlagInfo{"--limit", {"-l"}, "Limit"});
  ASSERT_TRUE(code.value()->add_flag(limit, diagnostic.emitter()).succeeded());
  auto result =
      run({"--verbose", "inspect", "code", "first", "-l=-7", "second", "--", "--help", ""});
  ASSERT_TRUE(result.succeeded()) << diagnostic.message();
  EXPECT_EQ(result.value().command, code.value());
  EXPECT_EQ(result.value().command->path(), "tool inspect code");
  EXPECT_EQ(result.value().positional, (std::vector<std::string>{"first", "second", "--help", ""}));
  EXPECT_EQ(limit->value(), -7);
  EXPECT_TRUE(global->value());
  EXPECT_TRUE(run({"inspect", "code", "--verbose"}).failed());
  EXPECT_NE(diagnostic.message().find("tool inspect code"), std::string::npos);
}

TEST_F(CliTest, SameSpellingInDifferentCommands) {
  auto a = root.add_command("a", diagnostic.emitter());
  auto b = root.add_command("b", diagnostic.emitter());
  ASSERT_TRUE(a.succeeded() && b.succeeded());
  auto left = std::make_shared<IntFlag>(FlagInfo{"--x", {}, ""});
  auto right = std::make_shared<IntFlag>(FlagInfo{"--x", {}, ""});
  ASSERT_TRUE(a.value()->add_flag(left, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(b.value()->add_flag(right, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(run({"a", "--x=5"}).succeeded());
  EXPECT_EQ(left->value(), 5);
  EXPECT_EQ(right->value(), 0);
  ASSERT_TRUE(run({"b", "--x=8"}).succeeded());
  EXPECT_EQ(left->value(), 5);
  EXPECT_EQ(right->value(), 8);
}

TEST_F(CliTest, SharedFlagUsesLastAssignmentAcrossScopes) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c"}, "Count"});
  auto child = root.add_command("child", diagnostic.emitter());
  ASSERT_TRUE(child.succeeded());
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(child.value()->add_flag(value, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(run({"--count=2", "child", "-c=3"}).succeeded());
  EXPECT_EQ(value->value(), 3);
  ASSERT_TRUE(run({"child", "--count=4", "-c=5"}).succeeded());
  EXPECT_EQ(value->value(), 5);
}

TEST_F(CliTest, RegisteringSameAddressIsANoopButSameNameIsAnError) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c"}, "Count"}, 7);
  ASSERT_TRUE(root.add_flags("Counts", {value, value}, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flags("Other section", {value}, diagnostic.emitter()).succeeded());
  ASSERT_EQ(root.groups().size(), 1u);
  EXPECT_EQ(root.groups()[0].description, "Counts");
  EXPECT_EQ(root.flags(), (std::vector<std::shared_ptr<Flag>>{value}));
  EXPECT_EQ(root.find_flag("-c"), value);
  EXPECT_EQ(value->value(), 7);
  auto duplicate = std::make_shared<IntFlag>(FlagInfo{"--count", {}, "Different flag"});
  EXPECT_TRUE(root.add_flag(duplicate, diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "tool: spelling '--count' for --count conflicts with --count");
}

TEST_F(CliTest, HelpSectionsPreserveOrderAndDoNotIntroduceScope) {
  auto help =
      std::make_shared<BoolFlag>(FlagInfo{"--help", {"-h"}, "Show help", ValueMode::Switch});
  auto count = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c"}, "Number of items"});
  auto output = std::make_shared<StringFlag>(FlagInfo{"--output", {"-o"}, "Output file"});
  auto child = root.add_command("child", diagnostic.emitter());
  ASSERT_TRUE(child.succeeded());
  ASSERT_TRUE(root.add_flag(help, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flags("Processing", {count, output}, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flags("Empty", {}, diagnostic.emitter()).succeeded());
  EXPECT_EQ(format_help(root), "Usage: tool [options] <command> [--] [arguments...]\n"
                               "\nOptions:\n  --help, -h\n      Show help\n"
                               "\nProcessing:\n  --count, -c VALUE\n      Number of items\n"
                               "  --output, -o VALUE\n      Output file\n"
                               "\nCommands:\n  child\n");
  EXPECT_EQ(format_help(*child.value()), "Usage: tool child [options] [--] [arguments...]\n");
  auto result = run({"-c=3", "-o=out.txt", "child", "input"});
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().positional, (std::vector<std::string>{"input"}));
  EXPECT_EQ(count->value(), 3);
  EXPECT_EQ(output->value(), "out.txt");
}

TEST_F(CliTest, HelpUsesCustomOrEmptyPositionalUsage) {
  root.set_positional_usage("INPUT...");
  auto launch = root.add_command("launch", diagnostic.emitter(), PositionalMode::AfterSeparator);
  ASSERT_TRUE(launch.succeeded());
  launch.value()->set_positional_usage("APP [ARGS...]");
  auto config = std::make_shared<StringFlag>(FlagInfo{"--config", {}, "Config"});
  ASSERT_TRUE(launch.value()->add_flag(config, diagnostic.emitter()).succeeded());
  auto status = root.add_command("status", diagnostic.emitter());
  ASSERT_TRUE(status.succeeded());
  status.value()->set_positional_usage("");
  EXPECT_EQ(format_help(root), "Usage: tool [options] <command> [--] INPUT...\n"
                               "\nCommands:\n  launch\n  status\n");
  EXPECT_EQ(format_help(*launch.value()), "Usage: tool launch [options] -- APP [ARGS...]\n"
                                          "\nOptions:\n  --config VALUE\n      Config\n");
  EXPECT_EQ(format_help(*status.value()), "Usage: tool status [options]\n");
}

TEST_F(CliTest, BatchCollisionsDiagnoseAcrossAndWithinSections) {
  auto first = std::make_shared<IntFlag>(FlagInfo{"--first", {"-f"}, ""});
  auto fresh = std::make_shared<IntFlag>(FlagInfo{"--fresh", {}, ""});
  auto conflict = std::make_shared<IntFlag>(FlagInfo{"--second", {"-f"}, ""});
  ASSERT_TRUE(root.add_flags("First", {first}, diagnostic.emitter()).succeeded());
  EXPECT_TRUE(root.add_flags("Second", {fresh, conflict}, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("conflicts with --first"), std::string::npos);
  EXPECT_EQ(root.find_flag("--fresh"), nullptr);
  auto same_name = std::make_shared<IntFlag>(FlagInfo{"--fresh", {}, ""});
  EXPECT_TRUE(root.add_flags("Third", {fresh, same_name}, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("repeated spelling '--fresh'"), std::string::npos);
  EXPECT_TRUE(root.add_flags("Fourth", {fresh, nullptr}, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("null flag"), std::string::npos);
  EXPECT_EQ(root.groups().size(), 1u);
  ASSERT_TRUE(root.add_flags("Mixed", {first, fresh, fresh}, diagnostic.emitter()).succeeded());
  EXPECT_EQ(root.flags(), (std::vector<std::shared_ptr<Flag>>{first, fresh}));
}

TEST_F(CliTest, LookupRetainsMetadataAndValueBeyondCommandLifetime) {
  std::shared_ptr<Flag> retained;
  std::weak_ptr<Flag> weak;
  {
    Command other{"other"};
    auto flag = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c"}, "Count"}, 3);
    weak = flag;
    ASSERT_TRUE(other.add_flag(flag, diagnostic.emitter()).succeeded());
    retained = other.find_flag("-c");
  }
  ASSERT_FALSE(weak.expired());
  EXPECT_EQ(retained->info().name, "--count");
  EXPECT_EQ(retained->info().aliases, (std::vector<std::string>{"-c"}));
  EXPECT_EQ(retained->info().description, "Count");
  EXPECT_EQ(retained->format_value(), "3");
  ASSERT_TRUE(retained->parse_and_set("9", diagnostic.emitter()).succeeded());
  EXPECT_EQ(retained->format_value(), "9");
  retained.reset();
  EXPECT_TRUE(weak.expired());
  static_assert(!std::is_copy_assignable_v<IntFlag>);
  static_assert(!std::is_move_assignable_v<IntFlag>);
}

TEST_F(CliTest, RegistrationRejectsAllCollisionDirections) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--first", {"-f"}, ""});
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  for (FlagInfo info :
       {FlagInfo{"--first", {}, ""}, FlagInfo{"-f", {}, ""}, FlagInfo{"--second", {"--first"}, ""},
        FlagInfo{"--second", {"-f"}, ""}, FlagInfo{"--second", {"-s", "-s"}, ""},
        FlagInfo{"--second", {"--second"}, ""}}) {
    EXPECT_TRUE(
        root.add_flag(std::make_shared<IntFlag>(std::move(info)), diagnostic.emitter()).failed());
  }
  EXPECT_EQ(root.flags().size(), 1u);
  EXPECT_EQ(root.find_flag("--second"), nullptr);
  ASSERT_TRUE(root.add_command("child", diagnostic.emitter()).succeeded());
  EXPECT_TRUE(root.add_command("child", diagnostic.emitter()).failed());
}

TEST_F(CliTest, InvalidDefinitionsAndNullFlagsFail) {
  for (const char *name : {"", "--", "-", "---x", "-long", "plain", "--a=b", "--two words"})
    EXPECT_TRUE(
        root.add_flag(std::make_shared<IntFlag>(FlagInfo{name, {}, ""}), diagnostic.emitter())
            .failed())
        << name;
  EXPECT_TRUE(root.add_flag(nullptr, diagnostic.emitter()).failed());
  for (const char *name : {"", "--x", "two words", "a=b"})
    EXPECT_TRUE(root.add_command(name, diagnostic.emitter()).failed());
}

TEST_F(CliTest, ValueGrammarAndOwnedStrings) {
  auto text = std::make_shared<StringFlag>(FlagInfo{"--text", {"-t"}, ""});
  auto toggle = std::make_shared<BoolFlag>(FlagInfo{"--toggle", {}, "", ValueMode::Switch});
  ASSERT_TRUE(root.add_flag(text, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flag(toggle, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(run({"--text=", "--toggle=false"}).succeeded());
  EXPECT_TRUE(text->value().empty());
  EXPECT_FALSE(toggle->value());
  ASSERT_TRUE(run({"--text", "--toggle"}).succeeded());
  EXPECT_EQ(text->value(), "--toggle"); // A required value consumes the next token.
  ASSERT_TRUE(run({"--text=--", "--toggle"}).succeeded());
  EXPECT_EQ(text->value(), "--");
  EXPECT_TRUE(toggle->value());
  EXPECT_TRUE(run({"--text", "--"}).failed());
  EXPECT_TRUE(run({"--text"}).failed());
  EXPECT_TRUE(run({"--toggle=1"}).failed());
  std::string argument = "--text=owned";
  const std::array<std::string_view, 1> args{argument};
  ASSERT_TRUE(parse(root, args, diagnostic.emitter()).succeeded());
  argument.assign("changed");
  EXPECT_EQ(text->value(), "owned");
  ASSERT_TRUE(text->parse_and_set(text->value(), diagnostic.emitter()).succeeded());
}

TEST_F(CliTest, FailureStopsImmediatelyWithoutRollback) {
  auto a = std::make_shared<IntFlag>(FlagInfo{"--a", {"-a"}, ""});
  auto b = std::make_shared<IntFlag>(FlagInfo{"--b", {}, ""});
  ASSERT_TRUE(root.add_flag(a, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(root.add_flag(b, diagnostic.emitter()).succeeded());
  EXPECT_TRUE(run({"--a=3", "--unknown", "--b=4"}).failed());
  EXPECT_EQ(a->value(), 3);
  EXPECT_EQ(b->value(), 0);
  ASSERT_TRUE(run({"--a=1", "-a=2"}).succeeded());
  EXPECT_EQ(a->value(), 2);
  EXPECT_TRUE(run({"--a=3", "-a=bad", "--b=4"}).failed());
  EXPECT_EQ(a->value(), 3);
  EXPECT_EQ(b->value(), 0);
}

TEST_F(CliTest, CommandSelectionDoesNotExecuteAndSeparatorStopsSelection) {
  ASSERT_TRUE(root.add_command("child", diagnostic.emitter()).succeeded());
  auto empty = run({});
  ASSERT_TRUE(empty.succeeded());
  EXPECT_EQ(empty.value().command, &root);
  EXPECT_TRUE(run({"unknown"}).failed());
  auto result = run({"--", "child", "--whatever"});
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().command, &root);
  EXPECT_EQ(result.value().positional, (std::vector<std::string>{"child", "--whatever"}));
}

TEST_F(CliTest, NestedCommandCanRequireSeparatorBeforePositionals) {
  auto launch = root.add_command("launch", diagnostic.emitter(), PositionalMode::AfterSeparator);
  ASSERT_TRUE(launch.succeeded());
  EXPECT_TRUE(run({"launch", "application"}).failed());
  EXPECT_NE(diagnostic.message().find("tool launch: expected --"), std::string::npos);
  auto result = run({"launch", "--", "application", "--flag"});
  ASSERT_TRUE(result.succeeded()) << diagnostic.message();
  EXPECT_EQ(result.value().command, launch.value());
  EXPECT_TRUE(result.value().saw_separator);
  EXPECT_EQ(result.value().positional, (std::vector<std::string>{"application", "--flag"}));
}

TEST(CliValues, LauncherRequiresSeparatorAndOwnsForwardedArguments) {
  Command command{"launcher", PositionalMode::AfterSeparator};
  util::StringDiagnostic diagnostic;
  auto config = std::make_shared<StringFlag>(FlagInfo{"--config", {}, ""});
  ASSERT_TRUE(command.add_flag(config, diagnostic.emitter()).succeeded());
  const std::array<std::string_view, 1> bare{"application"};
  EXPECT_TRUE(parse(command, bare, diagnostic.emitter()).failed());
  EXPECT_NE(diagnostic.message().find("expected --"), std::string::npos);
  const std::array<std::string_view, 1> empty_separator{"--"};
  auto empty = parse(command, empty_separator, diagnostic.emitter());
  ASSERT_TRUE(empty.succeeded());
  EXPECT_TRUE(empty.value().saw_separator);
  EXPECT_TRUE(empty.value().positional.empty());
  std::string name = "application";
  const std::array<std::string_view, 7> args{"--config", "file.json", "--", name,
                                             "--config", "",          "--"};
  auto result = parse(command, args, diagnostic.emitter());
  ASSERT_TRUE(result.succeeded());
  name.assign("changed");
  EXPECT_EQ(result.value().positional,
            (std::vector<std::string>{"application", "--config", "", "--"}));
  EXPECT_EQ(config->value(), "file.json");
}

TEST_F(CliTest, RegistrationSealsButValuesRemainWritable) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--x", {}, ""});
  auto child = root.add_command("child", diagnostic.emitter());
  ASSERT_TRUE(child.succeeded());
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(run({}).succeeded());
  EXPECT_TRUE(
      root.add_flag(std::make_shared<IntFlag>(FlagInfo{"--y", {}, ""}), diagnostic.emitter())
          .failed());
  EXPECT_TRUE(child.value()->add_command("late", diagnostic.emitter()).failed());
  EXPECT_TRUE(root.find_flag("--x")->parse_and_set("7", diagnostic.emitter()).succeeded());
  EXPECT_EQ(value->value(), 7);
}

TEST_F(CliTest, IndependentInstancesAndLifetime) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--x", {}, ""});
  std::weak_ptr<Flag> weak = value;
  {
    Command other{"tool"};
    ASSERT_TRUE(other.add_flag(value, diagnostic.emitter()).succeeded());
    value.reset();
    ASSERT_FALSE(weak.expired());
    auto retained = other.find_flag("--x");
    ASSERT_TRUE(retained->parse_and_set("9", diagnostic.emitter()).succeeded());
    EXPECT_EQ(root.find_flag("--x"), nullptr);
  }
  EXPECT_TRUE(weak.expired());
  auto retained = std::make_shared<StringFlag>(FlagInfo{"--x", {}, ""}, "kept");
  {
    Command other{"tool"};
    ASSERT_TRUE(other.add_flag(retained, diagnostic.emitter()).succeeded());
  }
  EXPECT_EQ(retained->value(), "kept");
}

TEST_F(CliTest, MetadataAndHelpRemainStableWhileRegistering) {
  auto value = std::make_shared<IntFlag>(FlagInfo{"--count", {"-c"}, "Item count"});
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  auto first = root.find_flag("--count");
  for (int i = 0; i < 100; ++i)
    ASSERT_TRUE(
        root.add_flag(std::make_shared<IntFlag>(FlagInfo{"--x" + std::to_string(i), {}, ""}),
                      diagnostic.emitter())
            .succeeded());
  EXPECT_EQ(root.find_flag("-c"), first);
  EXPECT_NE(format_help(root).find("--count, -c VALUE"), std::string::npos);
  EXPECT_NE(format_help(root).find("Item count"), std::string::npos);
  static_assert(!std::is_copy_constructible_v<Command>);
  static_assert(!std::is_move_constructible_v<Command>);
}

TEST(CliValues, IntegerBoundariesAndWholeInputConversion) {
  IntFlag signed_value({"--signed", {}, ""});
  UIntFlag unsigned_value({"--unsigned", {}, ""});
  util::StringDiagnostic diagnostic;
  for (const char *text : {"-9223372036854775808", "9223372036854775807", "0"}) {
    ASSERT_TRUE(signed_value.parse_and_set(text, diagnostic.emitter()).succeeded());
    EXPECT_EQ(signed_value.format_value(), text);
  }
  for (const char *text :
       {"", "9223372036854775808", "-9223372036854775809", "1x", " 1", "1 ", "+1", "0x10"})
    EXPECT_TRUE(signed_value.parse_and_set(text, diagnostic.emitter()).failed()) << text;
  ASSERT_TRUE(
      unsigned_value.parse_and_set("18446744073709551615", diagnostic.emitter()).succeeded());
  EXPECT_EQ(unsigned_value.value(), std::numeric_limits<uint64_t>::max());
  for (const char *text : {"18446744073709551616", "-1", "-0", ""})
    EXPECT_TRUE(unsigned_value.parse_and_set(text, diagnostic.emitter()).failed()) << text;
  EXPECT_TRUE(
      signed_value.parse_and_set(std::string_view("1\0x", 3), diagnostic.emitter()).failed());
}

class PositiveFlag final : public Flag {
public:
  using Flag::Flag;
  int64_t value() const { return value_; }
  Result set_value(int64_t value, DiagnosticEmitter diagnostic) {
    if (value <= 0)
      return diagnostic.emit() << "must be positive";
    value_ = value;
    return Result::success();
  }
  std::string format_value() const override { return std::to_string(value_); }
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override {
    auto parsed = parse_int(text, diagnostic);
    if (parsed.failed())
      return Result::failure();
    return set_value(parsed.value(), diagnostic);
  }

private:
  int64_t value_ = 1;
};

class AppendingFlag final : public Flag {
public:
  using Flag::Flag;
  const std::vector<std::string> &values() const { return values_; }
  std::string format_value() const override {
    std::string result;
    for (const std::string &value : values_) {
      if (!result.empty())
        result += ",";
      result += value;
    }
    return result;
  }
  Result parse_value(std::string_view text, DiagnosticEmitter) override {
    values_.emplace_back(text);
    return Result::success();
  }

private:
  std::vector<std::string> values_;
};

TEST_F(CliTest, CustomValidationSharesTypedAndStringPath) {
  auto value = std::make_shared<PositiveFlag>(FlagInfo{"--positive", {}, ""});
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  EXPECT_TRUE(value->set_value(0, diagnostic.emitter()).failed());
  EXPECT_TRUE(root.find_flag("--positive")->parse_and_set("0", diagnostic.emitter()).failed());
  EXPECT_TRUE(run({"--positive=0"}).failed());
  EXPECT_EQ(diagnostic.message(), "tool: --positive: must be positive");
  ASSERT_TRUE(run({"--positive=42"}).succeeded());
  EXPECT_EQ(value->value(), 42);
}

TEST_F(CliTest, CustomFlagCanAppendRepeatedValuesInOrder) {
  auto value = std::make_shared<AppendingFlag>(FlagInfo{"--path", {}, ""});
  ASSERT_TRUE(root.add_flag(value, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(run({"--path", "first", "--path=second", "--path", ""}).succeeded());
  EXPECT_EQ(value->values(), (std::vector<std::string>{"first", "second", ""}));
  EXPECT_EQ(value->format_value(), "first,second,");
}

TEST_F(CliTest, SilentCustomFailureStillGetsContext) {
  class FailingFlag final : public Flag {
  public:
    using Flag::Flag;
    std::string format_value() const override { return {}; }
    Result parse_value(std::string_view, DiagnosticEmitter) override { return Result::failure(); }
  };
  ASSERT_TRUE(
      root.add_flag(std::make_shared<FailingFlag>(FlagInfo{"--fail", {}, ""}), diagnostic.emitter())
          .succeeded());
  EXPECT_TRUE(run({"--fail=x"}).failed());
  EXPECT_EQ(diagnostic.message(), "tool: --fail: invalid value 'x'");
}

TEST_F(CliTest, RegistrationFromAParsingCallbackIsRejected) {
  class RegisteringFlag final : public Flag {
  public:
    explicit RegisteringFlag(Command &command) : Flag({"--register", {}, ""}), command_(command) {}
    std::string format_value() const override { return {}; }
    Result parse_value(std::string_view, DiagnosticEmitter diagnostic) override {
      return command_.add_command("late", diagnostic).failed() ? Result::failure()
                                                               : Result::success();
    }

  private:
    Command &command_;
  };
  ASSERT_TRUE(
      root.add_flag(std::make_shared<RegisteringFlag>(root), diagnostic.emitter()).succeeded());
  EXPECT_TRUE(run({"--register=x"}).failed());
  EXPECT_EQ(root.find_command("late"), nullptr);
  EXPECT_NE(diagnostic.message().find("registration is closed"), std::string::npos);
}

TEST(CliValues, DeterministicRoundTripsAndMalformedArguments) {
  std::mt19937_64 random(0xC11);
  util::StringDiagnostic diagnostic;
  Command command{"fuzz"};
  auto number = std::make_shared<UIntFlag>(FlagInfo{"--number", {"-n"}, ""});
  auto text = std::make_shared<StringFlag>(FlagInfo{"--text", {}, ""});
  ASSERT_TRUE(command.add_flag(number, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(command.add_flag(text, diagnostic.emitter()).succeeded());
  for (int iteration = 0; iteration < 4000; ++iteration) {
    const uint64_t expected = random();
    ASSERT_TRUE(number->set_value(expected).succeeded());
    UIntFlag decoded({"--decoded", {}, ""});
    ASSERT_TRUE(decoded.parse_and_set(number->format_value(), diagnostic.emitter()).succeeded());
    EXPECT_EQ(decoded.value(), expected);

    std::string bytes(random() % 48, '\0');
    for (char &byte : bytes)
      byte = static_cast<char>(random() & 0xff);
    ASSERT_TRUE(text->parse_and_set(bytes, diagnostic.emitter()).succeeded());
    EXPECT_EQ(text->format_value(), bytes);
    const std::array<std::string_view, 5> args{iteration % 2 ? "--number" : "--text", bytes,
                                               iteration % 3 ? "--" : "--unknown", "--number",
                                               "42"};
    bool emitted = false;
    auto sink = [&](std::string_view message) { emitted = !message.empty(); };
    auto parsed = parse(command, args, DiagnosticEmitter(sink));
    if (parsed.failed())
      EXPECT_TRUE(emitted);
    else
      EXPECT_EQ(parsed.value().positional, (std::vector<std::string>{"--number", "42"}));
  }
}

} // namespace
} // namespace util::cli
