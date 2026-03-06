import sys

# Prefix used so external services can reliably detect progress lines.
PROGRESS_PREFIX = "[MINER_PROGRESS]"


def emit_progress(phase: str, current: int, total: int) -> None:
    try:
        total = max(int(total), 1)
        current = max(0, min(int(current), total))
        percent = int(current / total * 100)
    except Exception:
        # Never let progress math break the miner.
        total, current, percent = 1, 0, 0

    line = f"{PROGRESS_PREFIX} phase={phase} current={current} total={total} percent={percent}"
    # Use stdout explicitly so it matches existing miner logs.
    print(line, flush=True, file=sys.stdout)
