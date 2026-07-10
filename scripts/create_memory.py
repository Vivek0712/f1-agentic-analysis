"""Create the AgentCore Memory store for the slow loop. Run once.

    python scripts/create_memory.py
    export MEMORY_ID=<printed id>

Two behaviors come from the configuration:
  raw events    every (payload, brief) turn is stored per agent per race
                session and is what get_last_k_turns reads for the
                previous-brief handoff
  summaries     the summary strategy maintains a rolling strategic
                summary per (agent, session) under
                /summaries/{actorId}/{sessionId}, consumed by the
                post-session tier

Events expire after 30 days; audit permanence lives in the S3 Object
Lock bucket, deliberately outside Memory.
"""

from bedrock_agentcore.memory import MemoryClient


def main():
    client = MemoryClient()
    memory = client.create_memory_and_wait(
        name="f1_race_briefs",
        description="Slow-loop agent brief continuity per race session",
        event_expiry_days=30,
        strategies=[{
            "summaryMemoryStrategy": {
                "name": "SessionStrategicSummary",
                "namespaces": ["/summaries/{actorId}/{sessionId}"],
            }
        }],
    )
    print(f"MEMORY_ID={memory['id']}")


if __name__ == "__main__":
    main()
