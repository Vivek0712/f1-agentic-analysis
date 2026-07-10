# AWS Deployment Architecture

The system separates a deterministic analysis layer from an agent reasoning
layer, and routes agent work into three latency tiers. Nothing on the
sub-second decision path touches an LLM.

```
                         +--------------------------------------------+
 live timing / FastF1 -> | Kinesis Data Streams -> Lambda (normalize) |
                         +---------------------+----------------------+
                                               |
                              S3 telemetry lake (Parquet, per session)
                                     |  Athena / Glue catalog
                                     v
                    +--------------------------------------+
                    | Deterministic layer (this repo)      |
                    | ECS Fargate task per session:        |
                    | stint fits, pit loss, undercuts,     |
                    | anomalies, counterfactuals           |
                    +---------+---------------+------------+
                              |               |
              slow loop (EventBridge,   post-session / cross-season
              30-120s cadence)          (Step Functions)
                              |               |
                    +---------v----+   +------v--------------------+
                    | AgentCore    |   | Bedrock Batch             |
                    | Runtime      |   | CreateModelInvocationJob  |
                    | (Strands,   |   | batch_input.jsonl -> S3   |
                    | Claude       |   | one record per agent      |
                    | Haiku)       |   +------+--------------------+
                    +---------+----+          |
                              |               |
                    +---------v---------------v----+
                    | Compliance gate (Lambda)     |
                    | deterministic rules engine   |
                    | audit records -> S3 Object   |
                    | Lock (immutable) + DynamoDB  |
                    +---------+--------------------+
                              |
                    briefs -> engineer console / dashboard (S3 + CloudFront)
```

## Tier routing

| Tier | Trigger | Runtime | Agents | Latency budget |
|---|---|---|---|---|
| slow_loop | EventBridge every 60s during session | Bedrock Converse via AgentCore Runtime | Stint Analyst, Rival Watcher, Deg Explainer, Compliance Guardian | < 15s per brief |
| post_session | Step Functions on session end | Bedrock Batch | Driver Coach, Race Reporter | < 1h |
| cross_season | Scheduled weekly / on demand | Bedrock Batch | Track Historian | < 24h |

## Design rules

1. Payload-in / brief-out. Agents receive fitted parameters and measured
   events as JSON. No tool access to raw telemetry. Every claim in a brief
   traces to a payload number, which is what the groundedness eval checks.
2. Compliance is deterministic. The rules engine (versioned code) validates
   every recommendation. The Compliance Guardian explains failures; it holds
   no authority to pass or overturn anything.
3. Advisory only. `auto_execute` fails closed in the rules engine. Humans
   act; agents brief.
4. Audit everything. Each brief is stored with its payload hash, model id,
   ruleset version, and verdicts in S3 Object Lock (compliance mode) with a
   DynamoDB index. A brief that cannot be replayed is a bug.

## Cost shape (order of magnitude)

Slow loop: 4 agents x 60 briefs/session x ~4K input / 400 output tokens on
Sonnet-class pricing lands in single-digit dollars per race session.
Post-session and cross-season run on Bedrock Batch at the 50% batch discount;
a full 24-race season re-analysis is a low-hundreds-of-dollars job. The
deterministic layer is the cheap part: one Fargate task per session.

## What is deliberately absent

- No agent on the pit-call path. Sub-second decisions belong to the
  precomputed strategy solvers teams already run.
- No fine-tuning. The payload contract plus system prompts carry the domain.
- No raw-data RAG. Retrieval over telemetry invites ungrounded synthesis;
  the deterministic layer is the retrieval.


## Slow-loop continuity: AgentCore Memory

The slow-loop app (src/f1agents/agents/agentcore_app.py) uses Amazon
Bedrock AgentCore Memory with the agent name as actor_id and the race
session as session_id.

- Short-term: get_last_k_turns fetches the agent's previous brief before
  each cadence tick, so briefs lead with deltas.
- Long-term: a summary strategy maintains a rolling strategic summary per
  (agent, session) under /summaries/{actorId}/{sessionId}; the
  post-session tier retrieves it for the Race Reporter payload.
- Retention: raw events expire after 30 days. Permanence is the S3
  Object Lock audit bucket, not Memory.
- Failure mode: every memory call degrades to stateless operation; a
  Memory outage costs continuity, never a brief.

Bootstrap once with scripts/create_memory.py and set MEMORY_ID on the
runtime.
