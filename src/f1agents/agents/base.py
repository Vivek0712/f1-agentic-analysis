"""Bedrock-backed analysis agents.

Each agent is a thin reasoning layer over deterministic payloads:
system prompt + structured JSON in, structured brief out. No agent has
tool access to raw telemetry; the contract is payload-in/brief-out so that
every claim in a brief is traceable to a number in the payload.

Runtime targets:
- In-race slow loop  -> Bedrock Converse (on-demand, AgentCore Runtime)
- Post-session       -> Bedrock Batch (one job per race, fleet of agents)
- Cross-season       -> Bedrock Batch (one job per driver/circuit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    import boto3
except ImportError:  # offline analysis does not need boto3
    boto3 = None

PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_MODEL = "anthropic.claude-sonnet-4-6"


@dataclass
class AgentSpec:
    name: str
    prompt_file: str
    latency_tier: str      # "slow_loop" | "post_session" | "cross_season"
    max_tokens: int = 1500

    @property
    def system_prompt(self) -> str:
        return (PROMPT_DIR / self.prompt_file).read_text()


AGENTS: dict[str, AgentSpec] = {
    "stint_analyst":       AgentSpec("Stint Analyst", "stint_analyst.md", "slow_loop"),
    "rival_watcher":       AgentSpec("Rival Watcher", "rival_watcher.md", "slow_loop"),
    "deg_explainer":       AgentSpec("Deg Explainer", "deg_explainer.md", "slow_loop"),
    "driver_coach":        AgentSpec("Driver Coach", "driver_coach.md", "post_session"),
    "race_reporter":       AgentSpec("Race Reporter", "race_reporter.md", "post_session", 2500),
    "track_historian":     AgentSpec("Track Historian", "track_historian.md", "cross_season", 2000),
    "compliance_guardian": AgentSpec("Compliance Guardian", "compliance_guardian.md", "slow_loop", 800),
    # dashboard-only: drill-down narratives (not part of the race-brief roster)
    "championship_strategist": AgentSpec("Championship Strategist", "championship_strategist.md", "cross_season", 1200),
    "season_analyst":          AgentSpec("Season Analyst", "season_analyst.md", "cross_season", 1200),
    "driver_review":           AgentSpec("Driver Season Review", "driver_review.md", "cross_season", 1200),
}


class BedrockAgent:
    def __init__(self, spec: AgentSpec, model_id: str = DEFAULT_MODEL,
                 region: str = "us-east-1"):
        self.spec = spec
        self.model_id = model_id
        self._client = None
        self._region = region

    @property
    def client(self):
        if self._client is None:
            if boto3 is None:
                raise RuntimeError("boto3 required for live agent runs")
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def run(self, payload: dict) -> str:
        """Single-turn brief generation from a deterministic payload."""
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": self.spec.system_prompt}],
            messages=[{
                "role": "user",
                "content": [{"text": json.dumps(payload, default=str)}],
            }],
            inferenceConfig={"maxTokens": self.spec.max_tokens, "temperature": 0.2},
        )
        return resp["output"]["message"]["content"][0]["text"]

    def batch_record(self, record_id: str, payload: dict) -> dict:
        """One JSONL record for a Bedrock Batch (CreateModelInvocationJob) run."""
        return {
            "recordId": record_id,
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "system": self.spec.system_prompt,
                "max_tokens": self.spec.max_tokens,
                "temperature": 0.2,
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
                }],
            },
        }
