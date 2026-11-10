"""Queue-switch and starvation diagnostics; log transitions rather than every tick."""

from __future__ import annotations


class QueueDiagnostics:
    def __init__(self, hz):
        self.period = 1 / hz
        self.hold_count = 0
        self.hold_ticks = 0
        self.hold_total_s = 0.0
        self.hold_started = None
        self.current_hold_ticks = 0

    def hold_tick(self, now, request_id):
        if self.hold_started is None:
            self.hold_started = now
            self.current_hold_ticks = 0
            self.hold_count += 1
            print(
                f"queue_hold_start count={self.hold_count} pending_chunk={request_id} old_remaining=0",
                flush=True,
            )
        self.current_hold_ticks += 1
        self.hold_ticks += 1

    def end_hold(self, now, reason):
        if self.hold_started is None:
            return
        duration = max(0.0, now - self.hold_started)
        self.hold_total_s += duration
        print(
            f"queue_hold_end reason={reason} duration_s={duration:.3f} "
            f"ticks={self.current_hold_ticks} hold_count={self.hold_count} "
            f"hold_total_s={self.hold_total_s:.3f}",
            flush=True,
        )
        self.hold_started = None
        self.current_hold_ticks = 0

    def switch_fields(self, now, old_remaining, new_remaining, *, bootstrap=False):
        if not bootstrap:
            self.end_hold(now, "new_chunk")
        overlap = min(old_remaining, new_remaining)
        return (
            f"old_remaining={old_remaining} old_remaining_ms={old_remaining * self.period * 1000:.0f} "
            f"new_remaining={new_remaining} overlap={overlap} "
            f"hold_count={self.hold_count} hold_total_s={self.hold_total_s:.3f}"
        )

    def finish(self, now):
        self.end_hold(now, "session_end")
        print(
            f"queue_summary hold_count={self.hold_count} hold_ticks={self.hold_ticks} "
            f"hold_total_s={self.hold_total_s:.3f}",
            flush=True,
        )
