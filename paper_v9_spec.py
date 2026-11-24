"""Specification curves for v3.5 (fig_spec_v35): paper_v8_spec plus start weekdays in every panel, leave-one-year-out in A,
and the rows of Tables 5B, 8 and 9 that were missing. Numbers from results/paper (v3, v3_extra, v5, v6, v7, v8)."""
import re
from pathlib import Path
src = (Path(__file__).parent / "paper_v8_spec.py").read_text()
src = src.replace('("ex-ante funding filter", 0.004, 0.22)]', '("ex-ante funding filter", 0.004, 0.22),\n  ("ex 2020", -0.251, -4.05), ("ex 2021", -0.299, -4.37), ("ex 2022", -0.275, -3.90), ("ex 2023", -0.182, -3.38), ("ex 2024", -0.296, -4.25), ("ex 2025", -0.233, -3.38)]')
src = src.replace('("90-day universe", 0.219, 3.47)]', '("90-day universe", 0.219, 3.47), ("perp-only 0.30%", 0.487, 4.66), ("tiered costs", 0.487, 4.66),\n  ("Tue", 0.460, 4.52), ("Wed", 0.414, 4.07), ("Thu", 0.445, 4.34), ("Fri", 0.445, 4.28), ("Sat", 0.458, 4.30), ("Sun", 0.543, 3.72)]')
src = src.replace('("maintenance 5%", 0.597, 2.23)]', '("maintenance 5%", 0.597, 2.23), ("maintenance 2.5%", 0.605, 2.26), ("w/o top 5% by funding", 0.067, 0.24), ("perp-only 0.30%", 0.613, 2.30),\n  ("Tue", 0.792, 2.67), ("Wed", 0.802, 2.68), ("Thu", 0.442, 1.41), ("Fri", 0.648, 1.98), ("Sat", 0.477, 1.49), ("Sun", 0.600, 2.12)]')
src = src.replace('fig_spec_v3.{e}', 'fig_spec_v35.{e}')
assert src.count("ex 2025") == 1 and src.count('("Sun"') == 2 and src.count("fig_spec_v35") == 1
exec(compile(src, "spec35", "exec"))
for t, r in panels.items(): print(t, len(r), "|t|>=1.96 and positive", sum(x[1] > 0 and abs(x[2]) >= 1.96 for x in r), "negative sig", sum(x[1] < 0 and abs(x[2]) >= 1.96 for x in r), "positive", sum(x[1] > 0 for x in r))
