"""Orchestrator for the benchmark agent with Skills V2."""

import json
import logging
import sys
import textwrap
import time
import traceback
import uuid
from typing import Optional

from google.adk import runners
from google.adk.agents import RunConfig, llm_agent
from google.adk.models.google_llm import Gemini
from google.adk.planners import built_in_planner
from google.adk.sessions.session import Session
from google.adk.tools.skill_tool import SecureBashTool, DEFAULT_SYSTEM_INSTRUCTION_V2
from google.genai import types

from . import swebench_environment, terminalbench_environment, utils, benchmark_tools
from .skills import definitions

logger = logging.getLogger(__name__)


# Updated instructions to encourage skill usage
SYSTEM_INSTRUCTIONS = """You are a software engineering agent solving a issue reported by a user.
You are working in the background and do not have the ability to discuss with the user.
Make your best attempt at implementing a solution, and use the `submit` tool when done.

""" + DEFAULT_SYSTEM_INSTRUCTION_V2


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
        
        # Initialize internal benchmark tools (implementation)
        self.benchmark_tools = benchmark_tools.BenchmarkTools(env, self.working_dir)

        # Register tools as Skills
        file_editor_skill = definitions.create_file_editor_skill(self.benchmark_tools)
        shell_skill = definitions.create_shell_skill(self.benchmark_tools)

        self.skill_tool = SecureBashTool(skills=[file_editor_skill, shell_skill])


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
                if sys.getsizeof(session_arg.state[key]) > utils.MAX_STATE_SIZE_BYTES:
                    session_arg.state[key] = "removed"
        return session_arg

    async def run(
        self, max_turns: int = 120, model_name: str = "gemini-2.5-flash"
    ) -> tuple[str | None, list]:
        """Runs the agent to solve the benchmark task."""
        start_time = time.time()

        # Since we are using InMemoryClient + FunctionScript, the code runs LOCALLY in the orchestrator process, 
        # but calls self.env.execute(), which runs inside the container.
        # So we definitely DO NOT need to copy these specific skills to /skills in the container 
        # because they are "remote" skills (running on host, acting on container).

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
        
        if model_name is None:
            model_name = "gemini-2.5-flash"

        agent_instance = llm_agent.LlmAgent(
            name="benchmark_agent_with_skills",
            model=Gemini(
                model=model_name,
                retry_options=types.HttpRetryOptions(
                    attempts=10,
                    exp_base=2,
                    initial_delay=1,
                    http_status_codes=[429, 499, 500, 503, 504],
                ),
            ),
            planner=built_in_planner.BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True, thinking_budget=-1
                )
            ),
            instruction=SYSTEM_INSTRUCTIONS,
            tools=[self.skill_tool, self.benchmark_tools.submit],
        )

        runner = runners.InMemoryRunner(agent=agent_instance)
        logger.info("Running agent with %d turns using SKILLS (InMemory).", max_turns)
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
