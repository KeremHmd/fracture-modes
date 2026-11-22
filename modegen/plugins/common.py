"""General-purpose plugins usable with any profile."""

import json
import time

from ..interfaces import Param, PostProcessor
from ..registry import register


@register("post", "common.event_log")
class EventLog(PostProcessor):
    DESCRIPTION = "Records every impact (in memory, and optionally as JSON lines)."
    PARAMS = {
        "path": Param(None, "append JSON lines to this file (None = memory only)"),
        "echo": Param(False, "also print each record"),
    }

    def setup(self):
        self.events = []

    def _log(self, ctx, result):
        a = result.analysis or {}
        rec = dict(
            wall_time=time.time(), sim_time=float(ctx.world.time), tool=result.tool,
            branch=result.branch, reason=result.reason,
            point=[float(x) for x in ctx.event.point],
            magnitude=float(ctx.event.magnitude), piece_depth=int(ctx.piece.depth),
            material_class=a.get("material_class"), sigma_vm=a.get("sigma_vm_max"),
            n_instances_after=len(ctx.world.active_instances),
        )
        self.events.append(rec)
        if self.echo:
            print("[event_log]", rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    on_fracture = _log
    on_plastic = _log
    on_none = _log
