"""Run the real autonomous-ML pipeline through successful evaluation."""

from __future__ import annotations

from e2e_common import (
    PipelineStatus,
    prepare_demo_dataset,
    print_stage_results,
    print_terminal_failure,
    print_trace_location,
    run_pipeline,
    start_a2a_server,
)


def main() -> int:
    dataset = prepare_demo_dataset()
    print(
        f"Dataset: {dataset.path} ({dataset.source}, "
        f"task={dataset.task_type}, target={dataset.target_column})"
    )
    server = start_a2a_server("e2e")
    try:
        outcome = run_pipeline(dataset, run_name="e2e")
        print_stage_results(outcome)
        print(f"\nPipeline run_id: {outcome.state.run_id}", flush=True)
        print_trace_location(outcome)
        print_terminal_failure(outcome)
        return 0 if outcome.state.status == PipelineStatus.SUCCESS else 1
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
