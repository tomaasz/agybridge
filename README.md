# hermes-agy-plugin

A secure external-process model-provider plugin that connects [Hermes Agent](https://github.com/NousResearch/hermes-agent) to the AGY CLI.

It provides two Hermes model-provider profiles:

- **AGY** — high-effort reasoning profile;
- **AGY Fast** — low-effort reasoning profile.

## Why this plugin

AGY remains an external CLI with its own authentication. Hermes remains the coordinator for tool availability, approvals, command safety, logging, and session state.

When a task needs a tool, AGY emits an OpenAI-shaped `tool_call` block. The plugin converts it to a standard Hermes tool call; Hermes then validates and executes it through its normal dispatcher. AGY does **not** receive independent terminal, filesystem, browser, network, or permission access.

## Requirements

- Hermes Agent with external-process model-provider support and `agent.acp_openai_bridge`;
- AGY CLI installed and authenticated according to its upstream documentation;
- Python 3.11+.

The default profiles request these model IDs. Override them in the plugin if your AGY installation exposes different names:

- `gemini-3.8-flash-high` for **AGY**;
- `gemini-3.8-flash-low` for **AGY Fast**.

## Install

```bash
git clone https://github.com/tomaasz/hermes-agy-plugin.git
mkdir -p ~/.hermes/plugins/model-providers
cp -R hermes-agy-plugin/plugins/model-providers/agy \
  ~/.hermes/plugins/model-providers/agy
```

Restart the Hermes gateway or start a new Hermes process after installation.

Select a profile in a new session:

```text
/model gemini-3.8-flash-high --provider agy
/model gemini-3.8-flash-low --provider agy-fast
```

## Security model

| Capability | Owner |
|---|---|
| AGY authentication | AGY CLI |
| Model reasoning | AGY CLI |
| Tool schema and tool dispatch | Hermes |
| Command approvals and policy | Hermes |
| Execution, logging, and tool results | Hermes |

The provider defaults to `--mode plan --sandbox`. Write mode is supported only by an explicit client construction and rejects paths that are not isolated Git worktrees. The public profiles do not enable write mode.

## Development

Run the contract tests without a live AGY installation:

```bash
python -m pytest -q -o 'addopts=' tests/test_agy_provider.py
```

The suite verifies that the plugin exposes both profiles, forwards Hermes tool schemas, returns standard tool calls, and rejects write mode outside an isolated worktree.

## Compatibility

This plugin uses Hermes' public ACP text bridge (`agent.acp_openai_bridge`). It intentionally does not patch Hermes core files. Compatibility is tested against Hermes Agent `0.21.2`; earlier releases may not provide the bridge.

## License

MIT. See [LICENSE](LICENSE).
