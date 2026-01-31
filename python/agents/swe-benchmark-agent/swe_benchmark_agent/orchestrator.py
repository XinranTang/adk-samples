"""Orchestrator for the benchmark agent."""

import json
import logging
import time
import traceback
import uuid
import sys
import textwrap

from google.adk import runners
from google.adk.agents import RunConfig, llm_agent
from google.adk.models.google_llm import Gemini
from google.adk.planners import built_in_planner
from google.genai import types

from . import swebench_environment, terminalbench_environment, utils, benchmark_tools

logger = logging.getLogger(__name__)

# Updated instructions to encourage skill usage
SYSTEM_INSTRUCTIONS = "You are a software engineering agent solving a issue reported by a user. You are working in the background and do not have the ability to discuss with the user. Make your best attempt at implementing a solution, and call the `submit` tool when done."


class Orchestrator:
    """Orchestrator for the benchmark agent."""

    def __init__(
        self,
        env: swebench_environment.SWEBenchEnvironment
        | terminalbench_environment.TerminalBenchEnvironment,
        benchmark_type: str = "swebench",
    ):
        """Initialize orchestrator with an environment.

        Args:
          env: Either SWEBenchEnvironment (swebench) or TerminalBenchEnvironment
            (terminalbench)
          benchmark_type: Type of benchmark ("swebench" or "terminalbench")
        """
        self.env = env
        self.benchmark_type = benchmark_type
        self.patch = None
        self.trajectory = []  # Store complete agent trajectory
        
        if self.benchmark_type == "terminalbench":
            self.working_dir = self.env.get_working_dir()
        else:
            self.working_dir = "/testbed"
            
        self.turn_count = 0
        
        # Initialize internal benchmark tools
        self.benchmark_tools = benchmark_tools.BenchmarkTools(env, self.working_dir)


    @classmethod
    def _remove_inline_data_binary(cls, session_arg: runners.Session) -> runners.Session:
        """Removes inline data binary in the session events content parts."""
        for event_item in session_arg.events:
            if event_item.content:
                for part in event_item.content.parts:
                    if part.inline_data and part.inline_data.data:
                        part.inline_data.data = b"Inline data removed."

        # Clear large values from the state as well.
        if session_arg.state:
            for key in list(session_arg.state):
                if sys.getsizeof(session_arg.state[key]) > utils.MAX_STATE_SIZE_BYTES:
                    session_arg.state[key] = "removed"

        return session_arg

    async def run(
        self, max_turns: int = 120, model_name: str = "gemini-2.5-flash"
    ) -> tuple[str | None, list]:
        """Runs the agent to solve the benchmark task.

        Args:
            max_turns: The maximum number of turns to allow the agent to run.
            model_name: The name of the Gemini model to use.

        Returns:
            tuple: (patch, trajectory) where trajectory contains all tool calls and
            responses
        """
        start_time = time.time()

        # Initialize trajectory metadata
        self.trajectory = [
            {
                "type": "metadata",
                "timestamp": start_time,
                "instance_id": self.env.instance.get("instance_id", "unknown"),
                "repo": self.env.instance.get("repo", "unknown"),
                "max_turns": max_turns,
                "problem_statement": self.env.instance.get(
                    "problem_statement", ""
                ),
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

        agent = llm_agent.LlmAgent(
            name="benchmark_agent",
            model=Gemini(
                model=model_name,
                retry_options=types.HttpRetryOptions(
                    attempts=10,
                    exp_base=2,
                    initial_delay=1,
                    http_status_codes=[
                        429,
                        499,
                        500,
                        503,
                        504,
                    ],  # Retry on these HTTP errors
                ),
            ),
            planner=built_in_planner.BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True, thinking_budget=-1
                )
            ),
            instruction=SYSTEM_INSTRUCTIONS,
            tools=[
                self.benchmark_tools.read_file,
                self.benchmark_tools.create_file,
                self.benchmark_tools.edit_file,
                self.benchmark_tools.run_shell_command,
                self.benchmark_tools.undo_last_edit,
                self.benchmark_tools.submit,
            ],
        )

        runner = runners.InMemoryRunner(agent=agent)
        logger.info("Running agent with %d turns.", max_turns)
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
                    # synchronize turn count to inline tools for verification prompt
                    self.benchmark_tools.turn_count = self.turn_count
                
                # Check if we should terminate due to successful submission
                if self.benchmark_tools.patch and self.benchmark_tools.patch.strip():
                    self.patch = self.benchmark_tools.patch
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
