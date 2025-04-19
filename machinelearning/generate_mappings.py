#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    # 1) point at your train/ folder
    repo_root   = Path(__file__).parent.resolve()
    train_dir   = repo_root / "dataset" / "train"
    output_path = repo_root / "species_mapping.json"

    if not train_dir.exists():
        print(f"❌ Train folder not found at {train_dir}")
        return

    # 2) list all sub‑directories (these are your class names)
    classes = [p.name for p in train_dir.iterdir() if p.is_dir()]
    classes.sort()  # ensure deterministic ordering

    # 3) write out as JSON array
    with output_path.open("w") as f:
        json.dump(classes, f, indent=2)

    print(f"✅ Wrote {len(classes)} class names to {output_path}")

if __name__ == "__main__":
    main()