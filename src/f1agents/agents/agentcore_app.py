"""Slow-loop agents on Bedrock AgentCore Runtime with AgentCore Memory.

One runtime app serves the four slow-loop agents. The orchestrator (an
EventBridge-triggered Lambda during a session) invokes it every 60
seconds per agent with the latest deterministic payload. AgentCore
Memory gives each agent continuity across cadence ticks:

  short-term  the previous brief for this (agent, race session) pair is
              retrieved with get_last_k_turns and passed alongside the
              payload, so briefs lead with what changed instead of
              restating the race
  long-term   a summary strategy rolls the session's briefs into a
              running strategic summary under
              /summaries/{actorId}/{sessionId}; the Race Reporter's
              post-session payload includes that summary retrieved by
              semantic query, which is how in-race context reaches the
              post-race narrative without replaying every event

Memory identifiers: actor_id is the agent name, session_id is the race
session (e.g. 2026_british_R). Create the memory store once with
scripts/create_memory.py and export MEMORY_ID.

Deploy:
    agentcore configure -e src/f1agents/agents/agentcore_app.py
    agentcore launch

The runtime execution role needs bedrock:InvokeModel, read-only access
to the results prefix of the telemetry lake, and the AgentCore Memory
data-plane actions. Nothing in this app can reach raw timing data.
"""

from __future__ import annotations

import json
import os

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .strands_roster import TIERS, build_agent, build_message

app = BedrockAgentCoreApp()
MEMORY_ID = os.environ.get("MEMORY_ID")

_agents: dict = {}
_memory = MemoryClient() if MEMORY_ID else None


def _get_agent(name: str):
    if name not in _agents:
        _agents[name] = build_agent(name)
    return _agents[name]


def _previous_brief(agent_name: str, session_id: str) -> str | None:
    if not _memory:
        return None
    try:
        turns = _memory.get_last_k_turns(
            memory_id=MEMORY_ID, actor_id=agent_name,
            session_id=session_id, k=1)
        for turn in turns:
            for msg in turn:
                if msg.get("role") == "ASSISTANT":
                    return msg["content"]["text"]
    except Exception:
        return None  # memory is an enhancement, never a dependency
    return None


def _record(agent_name: str, session_id: str, payload: dict, brief: str):
    if not _memory:
        return
    try:
        _memory.create_event(
            memory_id=MEMORY_ID, actor_id=agent_name,
            session_id=session_id,
            messages=[(json.dumps(payload), "USER"), (brief, "ASSISTANT")])
    except Exception:
        pass


def session_summary(session_id: str, query: str = "race strategy state",
                    actor_id: str = "stint_analyst") -> list[dict]:
    """Long-term memory for the post-session tier: the rolled-up session
    summary the Race Reporter payload embeds."""
    if not _memory:
        return []
    try:
        return _memory.retrieve_memories(
            memory_id=MEMORY_ID,
            namespace=f"/summaries/{actor_id}/{session_id}",
            query=query, top_k=3)
    except Exception:
        return []  # memory is an enhancement, never a dependency


@app.entrypoint
def invoke(event: dict) -> dict:
    """event: {"agent": str, "session_id": str, "payload": dict}"""
    agent_name = event["agent"]
    if TIERS.get(agent_name) != "slow_loop":
        return {"error": f"{agent_name} does not run on the slow loop; "
                         "post and cross tiers run as Bedrock Batch"}
    session_id = event.get("session_id", "adhoc")
    payload = event["payload"]

    previous = _previous_brief(agent_name, session_id)
    agent = _get_agent(agent_name)
    result = agent(build_message(payload, previous_brief=previous))
    brief = str(result)

    _record(agent_name, session_id, payload, brief)
    return {"agent": agent_name, "session_id": session_id,
            "brief": brief, "had_previous_brief": previous is not None}


if __name__ == "__main__":
    app.run()
