"""Strands Agents roster over the payload-in / brief-out contract.

Each agent is a Strands Agent whose system prompt is one of the seven
markdown contracts in prompts/. The contract survives the framework:
agents receive a structured JSON payload from the deterministic layer
as their message and return a short brief. They carry no tools, so
there is nothing for them to retrieve or compute beyond the payload,
and the groundedness evaluation stays mechanical.

Model routing by tier:
  slow_loop     Claude Haiku (latency and cost dominate a 60s cadence)
  post_session  Claude Sonnet (reasoning depth over a full result set)
  cross_season  Claude Sonnet via Bedrock Batch (see agents/base.py;
                batch jobs take raw Converse records, not Strands apps)
"""

from __future__ import annotations

import json
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

PROMPT_DIR = Path(__file__).parent / "prompts"

TIERS = {
    "stint_analyst": "slow_loop",
    "rival_watcher": "slow_loop",
    "deg_explainer": "slow_loop",
    "compliance_guardian": "slow_loop",
    "driver_coach": "post_session",
    "race_reporter": "post_session",
    "track_historian": "cross_season",
}

MODEL_BY_TIER = {
    "slow_loop": dict(model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
                      temperature=0.2, max_tokens=1024),
    "post_session": dict(model_id="anthropic.claude-sonnet-4-6-v1:0",
                         temperature=0.3, max_tokens=2048),
    "cross_season": dict(model_id="anthropic.claude-sonnet-4-6-v1:0",
                         temperature=0.3, max_tokens=2048),
}


def load_prompt(agent_name: str) -> str:
    return (PROMPT_DIR / f"{agent_name}.md").read_text()


def build_agent(agent_name: str, region: str | None = None) -> Agent:
    tier = TIERS[agent_name]
    cfg = MODEL_BY_TIER[tier]
    model = BedrockModel(region_name=region, **cfg)
    return Agent(name=agent_name, model=model,
                 system_prompt=load_prompt(agent_name),
                 tools=[])  # payload contract: no tools, by design


def build_message(payload: dict, previous_brief: str | None = None) -> str:
    """Payload message, optionally carrying the previous brief.

    The previous brief comes from AgentCore Memory and lets slow-loop
    agents lead with what changed since the last cadence tick without
    the orchestrator diffing payloads for them.
    """
    msg = {"payload": payload}
    if previous_brief:
        msg["previous_brief"] = previous_brief
        msg["instruction"] = ("Lead with what changed relative to "
                              "previous_brief. If nothing material "
                              "changed, say so in one line.")
    return json.dumps(msg)
