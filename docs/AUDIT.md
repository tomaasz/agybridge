# AGY plugin audit

Audit date: 2026-09-13. Baseline: `c6fbf94` (`fix: retry denied AGY actions through Hermes bridge`). Compatibility reference: public Hermes Agent 0.21.2 and AGY CLI 1.2.2.

## Findings and fixes

| ID | Problem | Severity | Reproduction | Cause | Fix | Regression test |
|---|---|---:|---|---|---|---|
| AGY-01 | `stream=True` returned a non-streaming completion | High | `create(stream=True)` | Flag ignored | Return Hermes bridge chunks | `test_partial_ndjson_stream_true_usage_and_unicode` |
| AGY-02 | Second denied action with text could be accepted | High | Two denied results with text | Denial checked only for empty text | Exactly one retry; second denial fails | `test_denied_action_retries_once_and_second_denial_falls_back` |
| AGY-03 | Child inherited every parent environment variable | High | Set an unrelated parent secret | `os.environ` inherited | Small allowlist plus explicit opt-in | `test_child_environment_is_restricted` |
| AGY-04 | Unknown, duplicate and malformed tool calls were accepted | High | Crafted `<tool_call>` blocks | Parser lacked current tool set | Strict shape, ID, name, args, duplicates and choice | `test_malformed_or_unsafe_tool_calls_fail_closed` |
| AGY-05 | Output and prompt had no bounds | High | Oversized output or args | Unbounded capture | Bounded reader and 8 MiB/4 MiB/64 KiB limits | `test_argument_and_prompt_limits` |
| AGY-06 | AGY write mode could bypass Hermes approvals | High | `write=True` | `accept-edits` exposed | Reject write mode; writes remain Hermes calls | `test_write_mode_is_rejected_even_for_git_directory` |
| AGY-07 | Empty, partial, invalid and repeated results looked successful | High | Omit/duplicate/mangle result event | Malformed lines silently skipped | Exactly one valid result required | `test_empty_partial_invalid_and_multiple_results_fail` |
| AGY-08 | Timeout had no controlled process-tree cleanup | Medium | Hung child | No explicit termination policy | New process group, SIGTERM then SIGKILL | `test_process_error_redaction_timeout_and_command_safety` |
| AGY-09 | Fast profile used high model when model omitted | Medium | Fast profile default request | One hard-coded default | Profile-specific default | `test_fast_profile_uses_fast_default_model` |
| AGY-10 | Usage was always zero | Medium | Result contains token counts | Fields discarded | Map AGY/OpenAI usage names | `test_partial_ndjson_stream_true_usage_and_unicode` |
| AGY-11 | Arbitrary args could override safety flags | Medium | `--dangerously-skip-permissions` | User args concatenated | Reject extra process arguments | `test_process_args_cannot_override_security_mode` |
| AGY-12 | README claimed a private/nonexistent API | High | Public installation | Wrong bridge contract | Public hook, bridge import, version gate | public Hermes smoke test |
| AGY-13 | Unrelated Google/Gemini credentials were forwarded automatically | High | Set `GOOGLE_API_KEY` in the parent | Prefix-based environment allowlist | Credential variables require explicit opt-in | `test_child_environment_is_restricted` |
| AGY-14 | A denied-action retry could consume twice the request timeout | Medium | Slow first and second attempts | Timeout reset per subprocess | One deadline shared by both attempts | `test_denied_retry_shares_the_request_deadline` |
| AGY-15 | `close()` stopped only the last concurrent child | Medium | Start two completions, then close | One active-process slot | Track and stop every active process | `test_close_stops_all_concurrent_children_and_prevents_reuse` |
| AGY-16 | Invalid UTF-8 was silently replaced | Medium | Emit malformed bytes before NDJSON parsing | Decoder used replacement characters | Reject invalid UTF-8 before parsing | `test_empty_partial_invalid_and_multiple_results_fail` |
| AGY-17 | A positional argument could select an AGY subcommand before safety flags | High | Configure `args=["mcp"]` | Only dash-prefixed arguments were rejected | Require a directly executable wrapper and reject all extra arguments | `test_process_args_cannot_override_security_mode` |

## Verification and dependency audit

The baseline suite had 4 tests and passed despite these defects. The final suite has 25 behavioral tests using real subprocess stubs, split NDJSON writes, hangs, non-zero exits, invalid UTF-8, large Unicode responses, concurrent shutdown and denial retries. CI never logs in to Gemini. The plugin has no runtime dependencies beyond Python's standard library and Hermes' public API; pytest is test-only. No secrets, tokens, cookies, private hosts or user data are stored. Child environment and stderr diagnostics are bounded and filtered.

## Performance decisions

- One process per completion matches AGY's request-scoped print interface; a persistent session needs a stable upstream protocol that is not published.
- Schemas are rendered only for the selected `tool_choice`; conversation JSON uses compact separators. No cache is used because Hermes changes schemas per turn.
- Reader threads avoid stderr/stdout deadlocks, while caps bound memory. Unicode and a 10,000-character response are regression-tested.

## Open items and roadmap

| Priority | Idea | Problem/value | Difficulty | Risk/security | Core change | Standalone | Recommendation |
|---|---|---|---|---|---|---|---|
| P0 | Capability negotiation | Detect AGY flags/output versions | M | Low; fail closed | No | Yes | First |
| P0 | Compatibility matrix CI | Catch Hermes/AGY drift | M | Low; no credentials | No | Yes | Nightly |
| P1 | Timeout profiles | Long tasks need different limits | S | Low; keep upper bound | No | Yes | Add validated settings |
| P1 | Graceful degradation messages | Explain fallback clearly | S | Low | No | Yes | Improve diagnostics |
| P1 | Versioned bridge adapter | Isolate ACP changes | M | Low; reject unknown | No | Yes | Add adapter version |
| P1 | Local opt-in metrics | Measure latency/denials | M | Low; disabled by default | No | Yes | Counters only |
| P2 | Parallel tool-call validation | Reduce independent call latency | M | Medium; dispatcher stays owner | Possibly | Yes | Coordinate with loop |
| P2 | Resumable AGY session | Avoid startup cost | L | High; protocol lifecycle | Likely | No | Wait for stable protocol |
| P2 | macOS/Windows fixtures | Cover process and encoding differences | M | Low | No | Yes | Hosted matrix |
| P2 | Signed release artifacts | Verify GitHub installs | M | Low | No | Yes | Checksums/Sigstore |
| P2 | Maintainer guide/fixtures | Faster review and incidents | S | Low | No | Yes | Add contribution docs |
| P3 | Structured image handoff | Preserve image context | L | Medium; no image logs | No | Yes | After capabilities |
