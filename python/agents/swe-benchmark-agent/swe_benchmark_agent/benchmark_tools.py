"""BenchmarkTools for interacting with the environment."""

import logging
import os
import posixpath
import shlex
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from . import swebench_environment, terminalbench_environment, utils

logger = logging.getLogger(__name__)


class BenchmarkTools:
    """Internal implementation of benchmark tools (formerly InlineToolWrapper)."""

    def __init__(
        self,
        env: swebench_environment.SWEBenchEnvironment
        | terminalbench_environment.TerminalBenchEnvironment,
        working_dir: str,
    ):
        self.env = env
        self.working_dir = working_dir
        self.last_edit_backup = {}
        self.benchmark_type = (
            "terminalbench"
            if isinstance(
                env, terminalbench_environment.TerminalBenchEnvironment
            )
            else "swebench"
        )
        # We need these for submit logic
        self.turn_count = 0
        self.num_submit_calls = 0
        self.patch = None

    def read_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int = 0,
    ) -> str:
        """Get the content of a file by reading a given range of lines.

        If possible, try to use this tool to only reach the lines that are relevant to the task.

        Note: this tool does not work for files outside the repo. For example `/dev/null` or `/bin/bash` will not work. It can return FileNotFoundError if a file does not exist, please use the bash tool to browse the list of files available in the repo first.

        Args:
            file_path: The path of the file to read relative to the project directory.
            start_line: The 1-indexed line number to start reading from.
            end_line: The inclusive 1-indexed line number to end reading at. By default, it will read 500 lines from the start line. Set to -1 to read the entire file. Use the feature of the whole file sparingly as it may lead to very large responses.

        Returns:
            A string containing the read file content, or an error message.
        """
        start_line = int(start_line)
        end_line = int(end_line)

        exit_code, file_content = self.env.execute(
            f"cat {shlex.quote(file_path)}"
        )

        if exit_code != 0:
            logger.error("Error: file %s not found or not readable.", file_path)
            return f"Error: file {file_path} not found or not readable."

        end_msg = ""
        file_content_lines = file_content.splitlines()

        file_content_lines = [
            f"{line_idx + 1}\t{line}"
            for line_idx, line in enumerate(file_content_lines)
        ]

        if end_line == -1:
            end_line = len(file_content_lines)
        elif end_line == 0:
            end_line = start_line + 500 - 1

        if start_line < 1:
            return f"Error: Start line {start_line} must be 1-indexed."

        if start_line > len(file_content_lines):
            return (
                f"Error: Start line {start_line} must be less than or equal to"
                " the total number of lines in the file:"
                f" {len(file_content_lines)}."
            )

        end_line = min(end_line, len(file_content_lines))

        preamble = (
            f"Showing the content of file {file_path} from line {start_line} to"
            f" line {end_line}. Lines are annotated with line numbers followed "
            f"by **one extra** tab. There are {len(file_content_lines)}"
            "  lines in total in this file:\n"
        )

        if start_line > 1:
            preamble += f"\n... lines 1-{start_line - 1} above omitted ...\n"

        content_to_show = "\n".join(
            file_content_lines[start_line - 1 : end_line]
        )

        lines_after_omitted = len(file_content_lines) - end_line
        if lines_after_omitted > 0:
            end_msg = (
                f"\n... lines {end_line + 1}-{len(file_content_lines)} below"
                " omitted ..."
            )

        output = preamble + content_to_show + end_msg

        return output

    def create_file(self, file_path: str, content: str) -> str:
        """Create a new file."""
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            dir_name = os.path.dirname(file_path)
            if dir_name:
                self.env.execute(f"mkdir -p {shlex.quote(dir_name)}")

            self.env.copy_to(
                tmp_file_path, os.path.join(self.working_dir, file_path)
            )

            return f"File {file_path} created successfully."
        except Exception as e:
            return f"Error creating file {file_path}: {e}"
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def edit_file(
        self,
        target_file: str,
        diff_file: str | None = None,
        diff: str | None = None,
    ) -> str:
        # pylint: disable=line-too-long
        """Edit a file in the codebase. Use this to make any edits to the codebase. Do not use this tool to create new files. Use the create_file tool for that.

        Args:
            target_file: The path of the file (relative to the project directory) to edit.
            diff_file: The path to a file containing the search-replace diff.
            diff: The search-replace diff content directly. One of diff_file or diff must be provided.
              (1) Each SEARCH/REPLACE block starts with: <<<<<<< SEARCH
              (2) Followed by a contiguous chunk of lines to search for in the existing source code
              (3) This is followed by a dividing line: =======
              (4) Then followed by the lines to replace into the source code.
              (5) The end of the replace block: >>>>>>> REPLACE
              (6) Keep *SEARCH/REPLACE* blocks concise. Include just the changing lines, and a few surrounding lines if needed to make a unique match. The tool will return an error if there are multiple matches.
              (7) To make multiple edits to the same file, stack a series of concise *SEARCH/REPLACE* blocks.
              (8) Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks. Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.
              (9) The search parts in different *SEARCH/REPLACE* blocks should not overlap.
              (10) Do not put any code lines outside of *SEARCH/REPLACE* blocks.

              Example:
              <<<<<<< SEARCH
                hello!
              =======
                hello world!
              >>>>>>> REPLACE
              <<<<<<< SEARCH
                another line to search for
              =======
                another line to replace
              >>>>>>> REPLACE

        Note: This tool automatically backs up the file before editing. You can use undo_last_edit to revert, but only the most recent edit per file is backed up.

        Returns:
            A message indicating whether the edit was applied successfully.
        """
        # pylint: enable=line-too-long
        try:
            if diff_file:
                exit_code, content = self.env.execute(f"cat {shlex.quote(diff_file)}")
                if exit_code != 0:
                     return f"Error: Diff file {diff_file} not found or not readable."
                diff = content
            
            if not diff:
                return "Error: No diff content provided. Please provide either diff_file or diff argument."

            if not (
                "<<<<<<< SEARCH" in diff
                and "=======" in diff
                and ">>>>>>> REPLACE" in diff
            ):
                return "Error: Invalid diff format. Missing SEARCH/REPLACE markers."

            file_path = target_file
            exit_code, file_content = self.env.execute(
                f"cat {shlex.quote(file_path)}"
            )
            if exit_code != 0:
                return f"Error: File {file_path} not found or not readable."

            if self.last_edit_backup is None:
                self.last_edit_backup = {}
            self.last_edit_backup[file_path] = file_content

            original_file_content = file_content
            blocks = diff.split("<<<<<<< SEARCH")
            for block in blocks:
                if "=======" not in block or ">>>>>>> REPLACE" not in block:
                    continue

                search_part, replace_part = block.split("=======")
                search_block = search_part
                replace_block = replace_part.split(">>>>>>> REPLACE")[0]

                if file_content.count(search_block) == 0:
                    return f"Error: Search string not found in {file_path}."
                if file_content.count(search_block) > 1:
                    return (
                        f"Error: Ambiguous search string in {file_path}. Please provide"
                        " more context."
                    )

                file_content = file_content.replace(search_block, replace_block)

            if original_file_content == file_content:
                return (
                    "Error: No changes were made to the file. "
                    "Please check your diff and try again."
                )

            output = self.create_file(file_path, file_content)
            if "created successfully." not in output:
                raise ValueError(output)

            result = f"Successfully edited file {file_path}."

        except Exception as e:  # pylint: disable=broad-exception-caught
            result = f"Error applying diff: {e}"
            logger.error(result)

        return result

    def undo_last_edit(self, file_path: str) -> str:
        """Undo the last edit to a file made by the edit_file tool.

        This tool only revert the last changes made by the edit_file tool and does not affect edits made by other means (e.g., shell commands).
        The tool can only undo the last edit to the file. You cannot go back to more than one previous edit.

        Args:
            file_path: The path of the file (relative to the project directory) to undo the last edit to.

        Returns:
            A message indicating whether the undo was successful.
        """
        if not self.last_edit_backup or file_path not in self.last_edit_backup:
            return (
                f"Error: Unable to undo the last edit to file: {file_path}\nThe file"
                " is not found, or the file has not been edited, or the file has"
                " been reverted previously."
            )

        last_edit_state = self.last_edit_backup.pop(file_path)
        self.create_file(file_path, last_edit_state)

        return f"Successfully reverted the last edit to file: {file_path}."

    def run_shell_command(
        self,
        cmd: str,
    ) -> str:
        """Run any shell commands using /bin/sh."""
        timeout_seconds = 120
        max_lines = 500

        cmd_with_timeout = (
            f"timeout {timeout_seconds}s /bin/sh -c {shlex.quote(cmd)}"
        )

        exit_code, output = self.env.execute(cmd_with_timeout, demux=True)

        if exit_code == utils.TIMEOUT_EXIT_CODE:
            return f"Error: The command timed out after {timeout_seconds} seconds."
        elif exit_code == utils.MEMORY_LIMIT_EXIT_CODE:
            return "Error: The command exceeded the memory limit"

        stdout, stderr = output
        formatted_output = ""
        is_output_truncated = False

        if stdout:
            stdout, is_out_truncated = utils.maybe_truncate_output(stdout, max_lines)
            formatted_output += f"Stdout:\n{stdout}\n"
            is_output_truncated |= is_out_truncated
        if stderr:
            stderr, is_err_truncated = utils.maybe_truncate_output(stderr, max_lines)
            formatted_output += f"Stderr:\n{stderr}\n"
            is_output_truncated |= is_err_truncated

        output_preamble = f"Command exited with status {exit_code}\n"
        if is_output_truncated:
            output_preamble += "There are truncated output.\n\n"

        result = output_preamble + formatted_output
        return result

    def _is_test_file(self, file_path: str) -> bool:
        """Whether a file seems to be a test file."""
        file_path = Path(file_path)
        return (
            any(part in ("test", "tests") for part in file_path.parts)
            or file_path.name.endswith("_test.py")
            or file_path.name.startswith("test_")
        )

    def _run_tests_internal(self) -> tuple[bool, str]:
        """Internal method to run tests for terminalbench (not exposed to agent)."""
        # Get the task directory from the environment
        task_dir = getattr(self.env, "task_dir", None)
        if not task_dir:
            return False, "Task directory not available in environment"

        task_dir = Path(task_dir)
        if not task_dir.exists():
            return False, f"Task directory not found: {task_dir}"

        # Check for run-tests.sh in task root
        run_tests_script = task_dir / "run-tests.sh"
        if not run_tests_script.exists():
            return False, f"run-tests.sh not found in {task_dir}"

        # Copy test files into the container AFTER agent execution
        tests_dir = task_dir / "tests"
        if not tests_dir.exists():
            return False, f"Tests directory not found: {tests_dir}"

        # Create /tests directory in container
        self.env.execute("mkdir -p /tests")

        # Copy ALL files from tests directory into the container
        for test_file in tests_dir.iterdir():
            if test_file.is_file():
                try:
                    self.env.copy_to(
                        str(test_file), f"/tests/{test_file.name}"
                    )
                    # Make shell scripts executable
                    if test_file.suffix == ".sh":
                        self.env.execute(
                            f"chmod +x /tests/{test_file.name}"
                        )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    return (
                        False,
                        f"Failed to copy test file {test_file.name}: {e}",
                    )

        # Copy run-tests.sh to the container
        try:
            run_tests_path_in_container = posixpath.join(
                self.working_dir, "run-tests.sh"
            )
            self.env.copy_to(
                str(run_tests_script), run_tests_path_in_container
            )
            self.env.execute(f"chmod +x {run_tests_path_in_container}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Failed to copy run-tests.sh: {e}"

        # Run run-tests.sh with TEST_DIR environment variable set
        full_command = (
            f"export TEST_DIR=/tests && bash {run_tests_path_in_container}"
        )

        exit_code, output = self.env.execute(full_command)
        all_passed = exit_code == 0

        results = f"Test script: run-tests.sh\nPassed: {all_passed}\nOutput:\n{output}\n"

        return all_passed, results

    def submit(self) -> str:
        """Submits the current solution and ends the interaction."""
        if self.benchmark_type == "terminalbench":
            _, test_results = self._run_tests_internal()
            self.patch = test_results  # Store test results for evaluation
            # Don't show test results to agent, just confirm submission
            result = "Submitted successfully."
        else:
            command = "git ls-files -z --others  --exclude-standard | xargs -0 git add -N && git diff --text HEAD"
            exit_code, output = self.env.execute(command)
            if exit_code != 0:
                result = f"Error: Failed to submit. Output:\\n{output}"
            else:
                changed_files_cmd = "git status --porcelain | awk '{print $2}'"
                _, changed_files_out = self.env.execute(changed_files_cmd)
                changed_files = changed_files_out.splitlines()

                num_edited_code_files = 0
                for file_path in changed_files:
                    if self._is_test_file(file_path):
                        continue
                    file_name = os.path.basename(file_path)
                    if file_name.endswith(".toml") or file_name.endswith(".ini") or file_name == "setup.py":
                        continue
                    num_edited_code_files += 1

                if num_edited_code_files == 0:
                    return (
                        "Observation: No meaningful existing code files were edited."
                        " Remember that the repository is guaranteed to have issues and"
                        " you MUST fix them."
                    )

                self.num_submit_calls += 1

                if (
                    self.num_submit_calls == 1
                    and self.turn_count < utils.MAX_VERIFICATION_TURN_COUNT
                ):
                    verification_prompt = textwrap.dedent(
                        f"""
                        You are trying to submit your work, but before that, please carefully verify that you have performed the following steps:
                        1. You have **thoroughly** tested your solution.
                        2. Regression tests: You have run existing related tests.
                        
                        Ideally use the `pytest_runner` skill to verify your fix.
                        You handled {self.turn_count} tool calls so far.
                        """
                    )
                    return verification_prompt

                self.patch = output
                result = f"Submitted successfully.\nTurns Taken: {self.turn_count}\nFiles Edited: {num_edited_code_files}"

        return result
