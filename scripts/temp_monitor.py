"""Log CPU temperature and load while a long CPU training run is in progress.

Windows exposes the ACPI thermal zone through the perf counter
`\\Thermal Zone Information(*)\\Temperature` (kelvin).  Sampling it costs a
PowerShell process per tick, so the interval is 60 s.

    python scripts/temp_monitor.py [interval_seconds] [output_csv]

The CSV is appended to, so the script can be restarted without losing history.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PS = [
    "powershell", "-NoProfile", "-Command",
    "(Get-Counter '\\Thermal Zone Information(*)\\Temperature' "
    "-ErrorAction SilentlyContinue).CounterSamples | "
    "ForEach-Object { $_.CookedValue }",
]
PS_LOAD = [
    "powershell", "-NoProfile", "-Command",
    "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
    "Where-Object {$_.Name -eq '_Total'}).PercentProcessorTime",
]


def _run(cmd) -> list[float]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    vals = []
    for tok in out.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def main() -> int:
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("temp_log.csv")
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "temp_c_max", "cpu_load_pct"])
        print(f"logging every {interval}s to {out}", flush=True)
        while True:
            temps = [v - 273.15 for v in _run(PS) if v > 200]
            load = _run(PS_LOAD)
            row = [datetime.now().strftime("%H:%M:%S"),
                   f"{max(temps):.1f}" if temps else "",
                   f"{load[0]:.0f}" if load else ""]
            w.writerow(row)
            f.flush()
            print("  ".join(row), flush=True)
            time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
