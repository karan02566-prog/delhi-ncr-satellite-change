import json
from pathlib import Path

m = json.loads(Path("data/labels/data_labels_split_manifest.json").read_text())

for split_name, patches in m["splits"].items():
    cols = [p["col"] for p in patches]
    rows = [p["row"] for p in patches]
    print(f"{split_name}: n={len(patches)}, col range=[{min(cols)}, {max(cols)}], row range=[{min(rows)}, {max(rows)}]")

# check column-stripe separation + buffer gap (patch_size=256)
splits = m["splits"]
names = list(splits.keys())
print("\nColumn boundaries per split:")
bounds = {}
for name in names:
    cols = [p["col"] for p in splits[name]]
    bounds[name] = (min(cols), max(cols))
    print(f"  {name}: {bounds[name]}")

# pairwise check: no col overlap, and gap >= 256 (patch_size) between adjacent splits
print("\nPairwise leakage check (expect gap >= 256 between every pair):")
sorted_by_start = sorted(bounds.items(), key=lambda x: x[1][0])
for i in range(len(sorted_by_start) - 1):
    name_a, (a_min, a_max) = sorted_by_start[i]
    name_b, (b_min, b_max) = sorted_by_start[i + 1]
    gap = b_min - a_max
    status = "PASS" if gap >= 256 else "FAIL"
    print(f"  {status}: {name_a} (max_col={a_max}) -> {name_b} (min_col={b_min}), gap={gap}px")