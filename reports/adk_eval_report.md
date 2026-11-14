# ADK Eval Report — Model Selection Agent

| Case | Status | Time (ms) | Notes |
|------|--------|-----------|-------|
| `classification_success` | ✅ PASS | 2957 | All checks passed |
| `regression_success` | ✅ PASS | 604 | All checks passed |
| `retry_with_feedback` | ✅ PASS | 623 | All checks passed |
| `bad_input_rejected` | ✅ PASS | 0 | All checks passed |
| `injection_blocked` | ✅ PASS | 2 | All checks passed |
| `classification_all_metrics` | ✅ PASS | 677 | All checks passed |

**Result:** 6/6 cases passed.

## Check Details

### `classification_success` — PASS
- ✓ `is_ModelSelectionResult`
- ✓ `run_id_matches`
- ✓ `has_3_candidates`
- ✓ `best_in_candidates`
- ✓ `model_type_valid`
- ✓ `has_f1_weighted`
- ✓ `has_accuracy`
- ✓ `model_uri_prefix`
- ✓ `retry_attempt_zero`

### `regression_success` — PASS
- ✓ `is_ModelSelectionResult`
- ✓ `has_3_candidates`
- ✓ `model_type_valid`
- ✓ `has_rmse`
- ✓ `has_mae`
- ✓ `model_uri_prefix`

### `retry_with_feedback` — PASS
- ✓ `is_ModelSelectionResult`
- ✓ `retry_attempt_is_1`
- ✓ `has_rmse`
- ✓ `has_3_candidates`

### `bad_input_rejected` — PASS
- ✓ `error_raised`
- ✓ `error_mentions_valid`

### `injection_blocked` — PASS
- ✓ `injection_blocked_before_training`
- ✓ `response_identifies_prompt_injection`

### `classification_all_metrics` — PASS
- ✓ `is_ModelSelectionResult`
- ✓ `has_3_candidates`
- ✓ `has_candidates_list`
- ✓ `create_run_fired`
- ✓ `log_params_fired`
- ✓ `log_metrics_fired`
- ✓ `query_best_run_fired`
- ✓ `register_model_fired`
- ✓ `all_5_mcp_actions`
- ✓ `has_f1_weighted`
- ✓ `has_accuracy`
