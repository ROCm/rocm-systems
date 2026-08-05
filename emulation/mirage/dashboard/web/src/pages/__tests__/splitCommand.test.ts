import { describe, expect, it } from "vitest";
import { splitCommand } from "../SessionDetailPage";

describe("splitCommand", () => {
  it("splits plain whitespace-separated tokens", () => {
    expect(splitCommand("echo hello world")).toEqual(["echo", "hello", "world"]);
  });

  it("keeps single-quoted segments intact (regression: /bin/sh -c 'echo hello')", () => {
    expect(splitCommand("/bin/sh -c 'echo hello'")).toEqual([
      "/bin/sh",
      "-c",
      "echo hello",
    ]);
  });

  it("keeps double-quoted segments intact and handles escapes", () => {
    expect(splitCommand('python -c "print(\\"hi\\")"')).toEqual([
      "python",
      "-c",
      'print("hi")',
    ]);
  });

  it("supports backslash-escaped spaces outside quotes", () => {
    expect(splitCommand("ls /tmp/with\\ space")).toEqual([
      "ls",
      "/tmp/with space",
    ]);
  });

  it("collapses runs of whitespace and trims", () => {
    expect(splitCommand("  a   b\tc\n")).toEqual(["a", "b", "c"]);
  });

  it("preserves empty quoted tokens", () => {
    expect(splitCommand("echo '' end")).toEqual(["echo", "", "end"]);
  });

  it("throws on an unterminated single quote", () => {
    expect(() => splitCommand("/bin/sh -c 'echo hello")).toThrowError(
      /unterminated single quote/,
    );
  });

  it("throws on an unterminated double quote", () => {
    expect(() => splitCommand('echo "oops')).toThrowError(
      /unterminated double quote/,
    );
  });
});
