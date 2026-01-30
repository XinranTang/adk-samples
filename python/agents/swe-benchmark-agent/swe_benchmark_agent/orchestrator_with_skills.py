
"""Orchestrator for the benchmark agent with Skills."""

import json
import logging
import os
import posixpath
import shlex
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from google.adk import runners
from google.adk.agents import RunConfig
from google.adk.sessions.session import Session
from google.genai import types
from google.adk.tools.skill_tool import SkillTool
from google.adk.skills.file_system_client import FileSystemClient

from . import agent, swebench_environment, terminalbench_environment

logger = logging.getLogger(__name__)

# Updated instructions to encourage skill usage
SYSTEM_INSTRUCTIONS = """You are a software engineering agent solving a issue reported by a user.
You are working in the background and do not have the ability to discuss with the user.
Make your best attempt at implementing a solution, and call the `submit` tool when done.

You have access to a set of Skills that provide specialized capabilities.
use `manage_skills` to discover available skills and read their instructions.


For file operations (reading, editing), use the `file_editor` skill.
For testing and linting, use `pytest` and `pylint` skills.
"""

TIMEOUT_EXIT_CODE = 124
MEMORY_LIMIT_EXIT_CODE = 137
MAX_VERIFICATION_TURN_COUNT = 40
MAX_STATE_SIZE_BYTES = 1024


def maybe_truncate_output(
    output: str,
    max_lines: int = 50,
    max_characters_per_line: int = 320,
) -> tuple[str, bool]:
    """Truncate the output by omitting the middle lines."""
    if max_lines <= 0:
        return output, False

    lines = output.splitlines(keepends=True)
    truncated_lines = []
    for line in lines:
        if len(line) > max_characters_per_line:
             num_truncated_chars = len(line) - max_characters_per_line
             ellipsis = f"(...line too long, truncated {num_truncated_chars} characters...)"
             len_prefix = (max_characters_per_line + 1) // 2
             len_suffix = max_characters_per_line // 2
             prefix = line[:len_prefix]
             suffix = line[len(line) - len_suffix :]
             truncated_lines.append(f"{prefix}{ellipsis}{suffix}")
        else:
            truncated_lines.append(line)
            
    lines = truncated_lines
    num_lines = len(lines)

    if num_lines <= max_lines:
        return "".join(lines), False

    half_max_lines = max_lines // 2
    omitted_lines = num_lines - max_lines

    truncated_output = (
        f"(Output too large with {num_lines} lines. Only show the first and last"
        f" {half_max_lines} lines)\n"
        + "".join(lines[:half_max_lines])
        + f"\n(... truncated {omitted_lines} lines ...)\n"
        + "".join(lines[-(max_lines - half_max_lines) :])
    )

    return truncated_output, True


class OrchestratorWithSkills:
    """Orchestrator for the benchmark agent with Skills."""

    def __init__(
        self,
        env: swebench_environment.SWEBenchEnvironment
        | terminalbench_environment.TerminalBenchEnvironment,
        benchmark_type: str = "swebench",
    ):
        """Initialize orchestrator with an environment."""
        self.env = env
        self.benchmark_type = benchmark_type
        self.patch = None
        self.trajectory = []
        self.num_submit_calls = 0
        self.turn_count = 0
        if self.benchmark_type == "terminalbench":
            self.working_dir = self.env.get_working_dir()
        else:
            self.working_dir = "/testbed"
            
        # Initialize Skills Client
        skills_path = Path(__file__).parent / "skills"
        if not skills_path.exists():
            logger.warning(f"Skills directory not found at {skills_path}")
            
        # Use standard FileSystemClient. Script execution is handled by run_shell_command in container.
        self.skills_client = FileSystemClient(str(skills_path))
        self.skill_tool = SkillTool(client=self.skills_client)

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

        if exit_code == TIMEOUT_EXIT_CODE:
            return f"Error: The command timed out after {timeout_seconds} seconds."
        elif exit_code == MEMORY_LIMIT_EXIT_CODE:
            return "Error: The command exceeded the memory limit"

        stdout, stderr = output
        formatted_output = ""
        is_output_truncated = False

        if stdout:
            stdout, is_out_truncated = maybe_truncate_output(stdout, max_lines)
            formatted_output += f"Stdout:\n{stdout}\n"
            is_output_truncated |= is_out_truncated
        if stderr:
            stderr, is_err_truncated = maybe_truncate_output(stderr, max_lines)
            formatted_output += f"Stderr:\n{stderr}\n"
            is_output_truncated |= is_err_truncated

        output_preamble = f"Command exited with status {exit_code}\n"
        if is_output_truncated:
            output_preamble += "There are truncated output.\n\n"

        return output_preamble + formatted_output

    def _run_tests_internal(self) -> tuple[bool, str]:
        """Internal method to run tests for terminalbench."""
        task_dir = getattr(self.env, "task_dir", None)
        if not task_dir:
            return False, "Task directory not available in environment"
        task_dir = Path(task_dir)
        if not task_dir.exists():
            return False, f"Task directory not found: {task_dir}"
        run_tests_script = task_dir / "run-tests.sh"
        if not run_tests_script.exists():
            return False, f"run-tests.sh not found in {task_dir}"
        tests_dir = task_dir / "tests"
        if not tests_dir.exists():
            return False, f"Tests directory not found: {tests_dir}"
        self.env.execute("mkdir -p /tests")
        for test_file in tests_dir.iterdir():
            if test_file.is_file():
                try:
                    self.env.copy_to(str(test_file), f"/tests/{test_file.name}")
                    if test_file.suffix == ".sh":
                        self.env.execute(f"chmod +x /tests/{test_file.name}")
                except Exception as e:
                    return False, f"Failed to copy test file {test_file.name}: {e}"
        try:
            run_tests_path_in_container = posixpath.join(self.working_dir, "run-tests.sh")
            self.env.copy_to(str(run_tests_script), run_tests_path_in_container)
            self.env.execute(f"chmod +x {run_tests_path_in_container}")
        except Exception as e:
            return False, f"Failed to copy run-tests.sh: {e}"
        full_command = f"export TEST_DIR=/tests && bash {run_tests_path_in_container}"
        exit_code, output = self.env.execute(full_command)
        all_passed = exit_code == 0
        results = f"Test script: run-tests.sh\nPassed: {all_passed}\nOutput:\n{output}\n"
        return all_passed, results

    def _is_test_file(self, file_path: str) -> bool:
        """Whether a file is a test file."""
        file_path = Path(file_path)
        return (
            any(part in ("test", "tests") for part in file_path.parts)
            or file_path.name.endswith("_test.py")
            or file_path.name.startswith("test_")
        )

    def submit(self) -> str:
        """Submits the current solution and ends the interaction."""
        if self.benchmark_type == "terminalbench":
            _, test_results = self._run_tests_internal()
            self.patch = test_results
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
                    and self.turn_count < MAX_VERIFICATION_TURN_COUNT
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

    @classmethod
    def _remove_inline_data_binary(cls, session_arg: Session) -> Session:
        """Removes inline data binary."""
        for event_item in session_arg.events:
            if event_item.content:
                for part in event_item.content.parts:
                    if part.inline_data and part.inline_data.data:
                        part.inline_data.data = b"Inline data removed."
        if session_arg.state:
            for key in list(session_arg.state):
                if sys.getsizeof(session_arg.state[key]) > MAX_STATE_SIZE_BYTES:
                    session_arg.state[key] = "removed"
        return session_arg

    async def run(
        self, max_turns: int = 120, model_name: str = "gemini-2.5-flash"
    ) -> tuple[str | None, list]:
        """Runs the agent to solve the benchmark task."""
        start_time = time.time()
        
        # Copy Skills to container
        try:
            skills_source = Path(__file__).parent / "skills"
            self.env.execute("mkdir -p /skills")
            
            if skills_source.exists():
                self.env.copy_to(str(skills_source), "/skills")
                self.env.execute("chmod -R +x /skills")
            else:
                logger.error("Skills directory missing on host!")
        except Exception as e:
            logger.error(f"Failed to copy skills to container: {e}")

        self.trajectory = [
            {
                "type": "metadata",
                "timestamp": start_time,
                "instance_id": self.env.instance.get("instance_id", "unknown"),
                "repo": self.env.instance.get("repo", "unknown"),
                "max_turns": max_turns,
                "problem_statement": self.env.instance.get("problem_statement", ""),
            }
        ]
        task_specification = (
            textwrap.dedent(
                f"""\
            I need you to solve the following issue in {self.env.instance["repo"]}:
            <issue>
            {{PROBLEM_STATEMENT}}
            </issue>
        """
            )
            .strip()
            .format(PROBLEM_STATEMENT=self.env.instance["problem_statement"])
        )

        # Since root_agent is a global singleton, we copy it or modify it. 
        # Modifying it is fine for single-threaded usage.
        agent_instance = agent.root_agent
        # Update model if needed
        if model_name != "gemini-2.5-flash":  # Only update if different from default
             agent_instance.model = Gemini(
                model=model_name,
                retry_options=types.HttpRetryOptions(
                    attempts=10,
                    exp_base=2,
                    initial_delay=1,
                    http_status_codes=[429, 499, 500, 503, 504],
                ),
            )
        
        agent_instance.tools = [
            self.skill_tool,
            self.run_shell_command,
            self.create_file,
            self.submit,
        ]

        runner = runners.InMemoryRunner(agent=agent_instance)
        logger.info("Running agent with %d turns using SKILLS.", max_turns)
        config = RunConfig(max_llm_calls=max_turns)
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="user",
            session_id=session_id,
        )

        self.turn_count = 0
        try:
            async for event in runner.run_async(
                user_id="user",
                session_id=session_id,
                run_config=config,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=task_specification)]
                ),
            ):
                if getattr(event.content, "role", None) == "model":
                    self.turn_count += 1
                if self.patch and self.patch.strip():
                    break
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(traceback.format_exc())

        current_session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id="user",
            session_id=session_id,
        )

        current_session = self._remove_inline_data_binary(current_session)
        self.trajectory.extend(
            json.loads(current_session.model_dump_json())["events"]
        )

        end_time = time.time()
        await runner.close()
        self.trajectory.append(
            {
                "timestamp": end_time,
                "duration": end_time - start_time,
                "total_turns": self.turn_count,
                "patch_generated": self.patch is not None,
            }
        )

        return self.patch, self.trajectory
