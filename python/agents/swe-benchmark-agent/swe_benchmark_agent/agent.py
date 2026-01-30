"""Agent definition for the benchmark agent."""

import os
from google.adk.agents import llm_agent
from google.adk.models.google_llm import Gemini
from google.adk.planners import built_in_planner
from google.adk.tools.skill_tool import DEFAULT_SYSTEM_INSTRUCTION
from google.genai import types

# Updated instructions to encourage skill usage
SYSTEM_INSTRUCTIONS = """You are a software engineering agent solving a issue reported by a user.
You are working in the background and do not have the ability to discuss with the user.
Make your best attempt at implementing a solution, and call the `submit` tool when done.

"""

root_agent = llm_agent.LlmAgent(
    name="benchmark_agent_with_skills",
    model=Gemini(
        model=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
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
    instruction=SYSTEM_INSTRUCTIONS + DEFAULT_SYSTEM_INSTRUCTION,
    tools=[],  # Tools will be injected by the orchestrator
)
