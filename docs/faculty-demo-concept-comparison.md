# Faculty Demo Concept Comparison

## Scope

This note compares the faculty `DEMO` examples with the current Autonomous ML Engineer Crew implementation. The faculty repository is useful as a set of small concept demonstrations. It is not a complete architecture that should be copied into this project.

Faculty source reviewed at commit [`424d379`](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO).

## Faculty demo map

### CrewAI notebooks

The 17 CrewAI notebooks progress from basic agents to production-oriented concerns:

| Demo | Main concept |
|---|---|
| 1 | Single agent, tool use, and a sequential researcher-writer-editor crew |
| 2 | Hierarchical process with a manager agent and planning |
| 3 | Pydantic structured task outputs |
| 4 | Custom synchronous and asynchronous tools |
| 5 | Memory storage and retrieval |
| 6 | Task output, token usage, and tool-failure inspection |
| 7 | Knowledge sources and retrieval |
| 8 | Skills as prompt instructions, distinct from callable tools |
| 9 | Agent reasoning and crew-level planning |
| 10 | Task, step, and agent callbacks |
| 11 | Checkpointing and resuming runs |
| 12 | CrewAI Flows with unstructured and Pydantic state |
| 13 | Event-bus processing and custom listeners |
| 14 | Observability with Langfuse |
| 15 | Prompt customization |
| 16 | Agent and crew fingerprints |
| 17 | CrewAI Enterprise hallucination guardrails |

Sources: [CrewAI demo directory](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/CrewAI), [Flow and state notebook](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/CrewAI/CrewAI_Demo12.ipynb), [callback notebook](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/CrewAI/CrewAI_Demo10.ipynb), [observability notebook](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/CrewAI/CrewAI_Demo14.ipynb).

### ADK examples

The ADK directory separates concepts into small runnable packages:

- `agent_types_demo`: `LlmAgent`, `SequentialAgent`, `ParallelAgent`, and `LoopAgent`.
- `multi_agent_patterns_demo`: coordinator, sequential pipeline, fan-out/gather, agent-as-tool, loop/refinement, review, and human approval patterns.
- `session_runner_demo`: `Runner`, `InMemorySessionService`, and session state changes.
- `structured_output_demo`: Pydantic `output_schema` and structured state handoffs.
- `tools_agent_demo`: function tools, tool context, agent transfer, and tool callbacks.
- `callback_demo`: agent lifecycle and before/after tool callbacks.
- `mcp_demo`: two MCP tools used by two ADK agents, coordinated by a `SequentialAgent` through shared state.
- `A2A`: a local root agent calling a remote A2A agent through its Agent Card, plus a `to_a2a` fact-checker example.
- `custom_agent`: a `BaseAgent` with custom orchestration around sequential and loop agents.
- `Deployment`: the ADK Cloud Run deployment workflow.
- `Eval`: an eval set, metric configuration, stored result, and an HTML result renderer.

Sources: [ADK demo directory](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK), [multi-agent patterns](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/multi_agent_patterns_demo), [MCP example](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/mcp_demo), [A2A examples](https://github.com/soharabhossain/Agentic_AI/tree/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/A2A), [deployment instructions](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/Deployment/ReadMe.md).

## Evaluation example

The faculty evaluation example defines a finance agent with three deterministic calculation tools. Its eval set contains six cases: one happy path and one invalid-input path for each tool. The configuration checks three different properties:

- `tool_trajectory_avg_score` at `1.0`, requiring the expected tool-call sequence.
- `response_match_score` at `0.8`, comparing surface text using ROUGE-style overlap.
- `final_response_match_v2` at `0.7`, using a model to judge semantic agreement.

The stored HTML report shows a **33.3 percent overall pass rate**: two of six cases pass every configured threshold. All six cases pass semantic `final_response_match_v2`, while several fail the stricter text-overlap metric and two fail exact tool trajectory. This is useful evidence that an agent can be semantically correct yet fail a brittle expected-text or expected-trajectory test. It also shows why negative cases must record whether the expected behavior is to call the validating tool or reject the request before that call.

Sources: [evaluation README](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/Eval/finance_assistant/ReadMe.md), [eval set](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/Eval/finance_assistant/metrics_demo_evalset_v1.evalset.json), [metric configuration](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/Eval/finance_assistant/test_config.json), [stored HTML report](https://github.com/soharabhossain/Agentic_AI/blob/424d3795fda803aa783d6868b22bddd84a26fcc9/DEMO/ADK/Eval/reports/eval_results_report.html), [official ADK evaluation guide](https://google.github.io/adk-docs/evaluate/).

## Comparison with this project

| Concept | Faculty demonstration | Current project evidence | Assessment |
|---|---|---|---|
| Shared context | CrewAI Flow state and ADK session state examples | `PipelineState` and typed stage results in `contracts/models.py`; `AutonomousMLEngineerFlow` in `orchestrator/flow.py` | Implemented more strictly than the demos because each stage result is validated |
| Agentic workflow | Sequential, hierarchical, parallel, loop, and review examples | Sequential stages, conditional quality gate, evaluator-optimizer retry loop, and bounded deployment policy | Appropriate patterns are implemented; parallel or hierarchical execution would add complexity without a clear benefit |
| Structured output | CrewAI `output_pydantic` and ADK `output_schema` examples | Six closed Pydantic contracts and final JSON/Markdown output | Implemented |
| MCP | One MCP server with two ADK-consumed tools | Two FastMCP servers: dataset profiling and MLflow experiment logging | Implemented; the tools are separated by responsibility rather than placed in one demo server |
| A2A | ADK root agent calls a remote ADK agent | CrewAI Flow discovers the ADK Agent Card and sends a JSON-RPC `SendMessage` request through `orchestrator/a2a_client.py` | Implemented and better aligned with the interoperability requirement because the boundary is CrewAI to ADK |
| Callbacks | Separate agent/tool callback demos and CrewAI task callbacks | Flow stage callbacks plus ADK agent, model, and tool callbacks | Implemented with broader coverage |
| Guardrails | Tool validation examples and an Enterprise hallucination guardrail notebook | CSV, target, path, prompt-injection, toxic-language, raw-row, MCP, A2A, and Pydantic fail-closed checks | Implemented without depending on the Enterprise-only notebook feature |
| Observability | CrewAI plus Langfuse notebook | OpenTelemetry/Langfuse trace across Flow, A2A, ADK, MCP, retries, and deployment; MLflow lineage | Implemented |
| Evaluation | ADK eval set with trajectory, text, and semantic metrics | Pytest contracts/failure paths, trace-structure tests, held-out ML metrics, and `scripts/benchmark.py` for runtime, tokens, and cost | Strong deterministic and system evaluation exists; an ADK-format evaluation artifact is still useful for teaching and targeted ADK-agent regression testing |
| Deployment | ADK command for Cloud Run with a service account | Local serving, Docker target, and a custom Cloud Run code path | Local path is demonstrated; live Cloud Run evidence remains the material gap |

## What to adopt

1. **Add a compact evaluation-set view.** Reuse the faculty report's case-list/detail idea, but show this project's pipeline scenarios, stage trajectory, terminal state, metric, runtime, token use, and cost.
2. **Define explicit evaluation cases.** Cover a successful regression run, successful classification run, retry then success, exhausted retries, malformed input, prompt injection, malformed MCP output, malformed A2A output, deployment-policy rejection, and prediction monitoring.
3. **Evaluate different properties separately.** Use exact assertions for Pydantic schemas, stage order, terminal states, and tool/protocol events. Use semantic judging only for natural-language explanations. Do not make the whole pipeline fail because two correct explanations use different wording.
4. **Expose the execution trajectory.** The learning interface should show the expected and observed path through guardrails, data preparation, MCP profiling, A2A model selection, MLflow MCP logging, evaluation, retry routing, deployment, and final result.
5. **Keep happy and failure cases together.** The faculty eval set usefully pairs valid and invalid inputs for every tool. Apply the same pairing to pipeline stages and guardrails.
6. **Document a repeatable command.** Provide one command that runs the evaluation suite and one command that renders the latest report.

## What not to copy

1. **Do not replace the existing CrewAI Flow with ADK workflow agents.** The requirement calls for both frameworks and the current boundary is already clear: CrewAI owns orchestration; ADK owns model selection.
2. **Do not add every demonstrated pattern.** Parallel agents, hierarchical crews, checkpointing, memory, knowledge stores, and human approval should be added only when they solve a project requirement. They are not automatic signs of a better agentic system.
3. **Do not adopt exact-response matching as the main quality signal.** The faculty report itself shows semantically correct answers failing the `0.8` text-overlap threshold.
4. **Do not claim ADK tool-trajectory evaluation covers internal Python calls automatically.** This project's custom `BaseAgent` calls model-selection code directly; its MCP actions are instrumented in the application but are not ordinary ADK `FunctionTool` events. Pipeline trajectory should therefore be checked through the project's callbacks and traces unless the agent is deliberately refactored to expose those operations as ADK tools.
5. **Do not copy the demo's public Cloud Run choice without a security decision.** The faculty deployment walkthrough permits unauthenticated access for teaching. The project must state whether its final prediction endpoint is public, protected by an API key, or protected by Cloud Run IAM.
6. **Do not use the CrewAI Enterprise hallucination guardrail as a required dependency.** The current local guardrails and Pydantic validation satisfy the project need without introducing a paid feature.

## Recommended evaluation layers

The project should present evaluation as four layers rather than one score:

1. **Contract evaluation:** Pydantic validation, schema closure, state transitions, and terminal-state invariants.
2. **Agent and protocol evaluation:** callback sequence, A2A Agent Card and `SendMessage`, MCP call/return validation, retry feedback, and trace-parent relationships.
3. **ML evaluation:** held-out task metric, acceptance threshold, candidate comparison, retry behavior, and reproducibility.
4. **System evaluation:** end-to-end success rate, failure-path correctness, p50/p95 runtime, token use, estimated cost, deployment health, and prediction-monitoring alerts.

An optional fifth layer can use an ADK eval set for the model-selection agent's JSON request/response behavior. That layer should not replace the existing deterministic tests and end-to-end benchmark.
