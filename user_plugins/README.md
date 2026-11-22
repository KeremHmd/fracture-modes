# user_plugins

Every `*.py` file in this folder (names starting with `_` are skipped) is
imported automatically by every modegen tool, after the built-in plugins.
More folders can be added with the `MODEGEN_PLUGIN_PATH` environment variable
(`;`-separated on Windows) or with `--plugin-dir` on any tool.

See `example_ring_impact.py` for a complete, minimal plugin and the README's
plugin section for the category contracts.
