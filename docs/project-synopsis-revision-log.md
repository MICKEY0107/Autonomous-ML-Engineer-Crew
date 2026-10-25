# Project Ideation and Synopsis Revision Log

Date: 20 October 2026

## Files produced

- `Agentic_AI_Project_Synopsis_Revised.docx`
- `Autonomous ML Engineer Crew - Project Explainer Revised.docx`
- `architecture_diagram_publication.svg` (editable master)
- `architecture_diagram_publication_aligned.png` (rendered preview)

The original synopsis, explainer, architecture diagram, presentation, course handout, and template remain unchanged.

## Content kept unchanged

- Project title and objective
- Intended users and user-facing outcome
- Five established roles: Data-Prep Agent, Model-Selection and Tuning Agent, Evaluation Agent, Deployment and Observability Agent, and Orchestrator
- Core responsibilities of every agent
- Supervised classification and regression scope
- CSV, target-column, and prediction-goal inputs
- Evaluation-to-tuning feedback loop with bounded retries
- Team members, primary ownership areas, timeline dates, and responsibility matrix
- Planned trained model, model report card, `/predict` API, monitoring, and reusable pipeline

## Changes by artifact

| Artifact and section | Change made | Reason |
|---|---|---|
| Synopsis Section 4 | Preserved the original objectives and added the required Pydantic JSON and Markdown final-output format. | Satisfies the structured-output criterion without changing the product goal. |
| Synopsis Section 5 Agents and Responsibilities | Kept all five agents and their responsibilities. Added an explicit framework assignment: CrewAI for the orchestrator and surrounding agents, Google ADK for the existing model-selection specialist. | Makes actual use of both required frameworks clear. |
| Synopsis Section 5 Framework Boundary | Added a real A2A boundary between the CrewAI Orchestrator and the ADK Model-Selection and Tuning Agent. | Ordinary handoffs do not prove A2A interoperability. |
| Synopsis Section 5 Shared State | Added `PipelineState` and five Pydantic handoff or result models with clear producers and consumers. | Provides testable context sharing and type validation. |
| Synopsis Section 5 MCP Integration | Changed MCP from optional to mandatory and defined the Dataset Profile and MLflow Experiment MCP tools. | The requirement calls for at least two MCP tools. |
| Synopsis Section 5 Workflow | Preserved the original sequence and bounded retry loop while adding typed state, A2A, MCP, success, and terminal-failure behavior. | Makes the existing agentic patterns executable and verifiable. |
| Synopsis Section 5 Callbacks | Added stage callbacks and ADK agent, model, and tool callbacks with structured event fields. | Covers intermediate outputs, execution flow, validation failures, retries, and errors. |
| Synopsis Section 5 Security | Added file, prompt-injection, abusive-input, tool-return, output, secret, authorization, and raw-data guardrails. | Meets the security and callback-validation criterion. |
| Synopsis Section 5 Observability | Added Langfuse with OpenTelemetry plus a shared run ID and MLflow lineage. | Supplies tracing, auditability, and provenance across both frameworks and protocols. |
| Synopsis Section 5 Evaluation | Added repeatable end-to-end quality, reliability, p50 and p95 runtime, token, and cost measurements. | Meets the performance, cost, and runtime requirement. |
| Synopsis Section 5 Deployment | Selected Google Cloud Run and defined deployment evidence, health checks, authentication, Secret Manager, smoke tests, and rollback evidence. | Turns the deployment goal into a verifiable cloud plan. |
| Synopsis Sections 6 to 9 | Preserved resources, team roles, dates, and responsibilities while adding the missing compliance deliverables to their existing work. | Keeps fixed project decisions intact. |
| Synopsis Section 10 | Added an eleven-row verification plan that names the evidence required before claiming compliance. | Prevents design intent from being confused with implemented proof. |
| Project Explainer | Retained the original story, users, outputs, agents, scope, and retry loop. Added plain-language implementation contracts and a definition-of-done table. | Makes the refined ideation understandable during review and viva. |
| Architecture Diagram | Kept the same five agents and pipeline objective. Added typed state, two MCP tools, the CrewAI-to-ADK A2A boundary, callbacks and guardrails, Langfuse and OpenTelemetry, evaluation evidence, and Cloud Run. The final correction makes the CrewAI Orchestrator the A2A client, routes all typed stage results through `PipelineState`, places the final `PipelineResult` under Orchestrator ownership, and represents safeguards and telemetry as system-wide concerns. | Aligns the visual architecture with the revised written design and removes misleading point-to-point ownership. |

## Files deliberately not changed

- `Agentic_AI_Project_Synopsis.docx`
- `Autonomous ML Engineer Crew — Project Explainer Updated.docx`
- `architecture_diagram.svg`
- `Autonomous ML Engineer Crew - Presentation - Revised.pptx`

The presentation still reflects the earlier design and should be synchronized with the revised documents before the final presentation.

## Important limitation

These revisions make the ideation and synopsis cover all eleven requirements at the design and acceptance-criteria level. They do not constitute implementation proof. Full compliance still requires source code, tests, traces, benchmark results, and a working cloud deployment.
