##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Input, Static


class Terimnal(Container):
    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.current_directory = Path.cwd()
        self.output_text: str = ""
        self.input_text: str = ""
        self.input_prompt: str = ""
        self.has_focus: bool = True

        # Command history
        self.command_history: list[str] = []
        self.history_index: int = -1
        self.current_command: str = ""

        # Tab completion
        self.tab_completions: list[str] = []
        self.tab_index: int = -1
        self.tab_prefix: str = ""

    def compose(self) -> ComposeResult:
        # Output area with scroll wrapper
        with Vertical():
            with VerticalScroll(id="term-output-scroll"):
                yield Static("", id="terminal-output")
            yield Input(id="terminal-input", placeholder="")

    def on_mount(self) -> None:
        """Initialize the terminal."""
        # Update status
        self.add_output(
            f"Support quick/simple terminal commands.\ncwd: {self.current_directory}\n"
        )

        # Update the prompt
        self.update_prompt()
        # self.query_one("#terminal-input").focus()

    def update_prompt(self) -> None:
        """Update the command prompt in the input field."""
        input_widget = self.query_one("#terminal-input")
        current_path = Path(self.current_directory).name or self.current_directory

        if platform.system() != "Windows":
            prompt = f"{current_path} $ "
        else:
            prompt = f"{current_path}> "

        input_widget.placeholder = prompt

    def add_output(self, text: str) -> None:
        """Add text to the terminal output."""
        self.output_text += text
        scroll = self.query_one("#term-output-scroll", VerticalScroll)
        scroll.scroll_end(animate=False)

        # Ensure scroll to bottom
        scroll = self.query_one("#term-output-scroll")
        scroll.scroll_end(animate=False)

    def action_clear(self) -> None:
        """Clear the terminal output."""
        self.output_text = ""
        output = self.query_one("#terminal-output", Static)
        output.update(Text.from_ansi(""))

    def action_interrupt(self) -> None:
        """Interrupt the current process if any."""
        if self.current_process is not None:
            try:
                if platform.system() == "Windows":
                    import signal

                    self.current_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.current_process.terminate()
                self.add_output("\n^C\n")
            except Exception as e:
                self.add_output(f"\nFailed to interrupt process: {str(e)}\n")
            finally:
                self.current_process = None
        else:
            # If no process is running, just show ^C and clear the input
            self.add_output("\n^C\n")
            self.query_one("#terminal-input", Input).value = ""

    def run_command(self, command: str) -> None:
        """Run a system command and display its output."""
        # Add command to history
        if command.strip() and (
            not self.command_history or command != self.command_history[-1]
        ):
            self.command_history.append(command)
            self.history_index = len(self.command_history)

        # Show the command in the output
        prompt = self.query_one("#terminal-input", Input).placeholder
        self.add_output(f"{prompt}{command}\n")

        if not command.strip():
            return

        # Handle built-in commands
        if command == "clear" or command == "cls":
            self.action_clear()
            return
        elif command == "exit" or command == "quit":
            if hasattr(self.app, "exit"):
                self.app.exit()
            return
        elif command.startswith("cd "):
            try:
                path = command[3:].strip()
                if not path:
                    # Just "cd" usually goes to home directory
                    new_path = Path.home()
                else:
                    new_path = Path(path)

                    # Handle relative paths
                    if not new_path.is_absolute():
                        new_path = Path(self.current_directory) / new_path

                # Resolve any symlinks and normalize the path
                new_path = new_path.resolve()

                # Change to the new directory
                new_path_str = str(new_path)
                os.chdir(new_path_str)
                self.current_directory = new_path_str
                self.add_output(f"Changed directory to {self.current_directory}\n")
                self.update_prompt()
            except Exception as e:
                self.add_output(f"Error: {str(e)}\n")
            return

        # Execute the command with shell=True to support pipes and redirections
        try:
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                cwd=self.current_directory,
                text=True,
            )

            stdout, stderr = self.current_process.communicate()

            if stdout:
                self.add_output(stdout)
            if stderr:
                self.add_output(f"{stderr}")

            self.current_process = None
        except Exception as e:
            self.add_output(f"Error executing command: {str(e)}\n")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "terminal-input":
            command = event.input.value
            event.input.value = ""

            self.run_command(command)

            event.input.focus()

    def on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation."""
        # Handle arrow keys for command history
        if event.key == "up" and self.command_history:
            input_widget = self.query_one("#terminal-input", Input)
            if self.history_index > 0:
                self.history_index -= 1
                input_widget.value = self.command_history[self.history_index]
                input_widget.cursor_position = len(input_widget.value)
            event.prevent_default()

        elif event.key == "down" and self.command_history:
            input_widget = self.query_one("#terminal-input", Input)
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                input_widget.value = self.command_history[self.history_index]
            else:
                self.history_index = len(self.command_history)
                input_widget.value = ""
            input_widget.cursor_position = len(input_widget.value)
            event.prevent_default()
