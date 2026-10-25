"""Run the pipeline with an intentionally impossible evaluation gate."""

from __future__ import annotations

from e2e_common import (
    PipelineStatus,
    prepare_demo_dataset,
    print_retry_spans,
    print_stage_results,
    print_terminal_failure,
    print_trace_location,
    run_pipeline,
    start_a2a_server,
)


FORCED_EVALUATION_THRESHOLD = 0.99


def main() -> int:
    dataset = prepare_demo_dataset()
    print(
        f"Dataset: {dataset.path} ({dataset.source}, "
        f"task={dataset.task_type}, target={dataset.target_column})"
    )
    print(f"Forced evaluation threshold: {FORCED_EVALUATION_THRESHOLD}")
    server = start_a2a_server("forced_fail")
    try:
        outcome = run_pipeline(
            dataset,
            run_name="forced_fail",
            evaluation_threshold=FORCED_EVALUATION_THRESHOLD,
        )
        print_stage_results(outcome)
        print_trace_location(outcome)
        print_terminal_failure(outcome)
        span_count = print_retry_spans(outcome)
        expected = (
            outcome.state.status == PipelineStatus.FAILED_EVAL_EXHAUSTED
            and outcome.state.retry_count == 2
            and span_count == 3
        )
        if not expected:
            print(
                "Forced-failure demonstration did not reach the expected "
                "FAILED_EVAL_EXHAUSTED state with three attempts."
            )
            return 1
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
