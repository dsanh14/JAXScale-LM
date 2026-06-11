#!/usr/bin/env bash
# Verify the repository works from TRACKED FILES ONLY.
#
# Guards against gitignore rules silently excluding source code (a real
# incident: an unanchored `data/` rule kept src/jaxscale_lm/data/ out of
# git, so the package only worked in checkouts that already had the files
# on disk). The check exports exactly what git tracks into a temp
# directory, installs the project there from the lock file, and imports
# the modules a fresh clone needs.
#
# Usage: scripts/check_clean_export.sh   (or: make check-export)
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

tmp="$(mktemp -d -t jaxscale-clean-export)"
trap 'rm -rf "$tmp"' EXIT

# Tracked files only (the index): catches ignored-but-needed sources both
# before a commit (with intent-to-add) and after (== committed set in CI).
# Entries deleted from the worktree (pending deletions) are skipped.
git ls-files -z | python3 -c '
import os, shutil, sys

dst = sys.argv[1]
count = 0
for raw in sys.stdin.buffer.read().split(b"\0"):
    if not raw:
        continue
    path = raw.decode()
    if not os.path.isfile(path):
        continue  # tracked but deleted in the worktree
    target = os.path.join(dst, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(path, target)
    count += 1
print(f"clean-export: exported {count} tracked files")
' "$tmp"

cd "$tmp"

uv sync --frozen --no-dev --quiet
# macOS: a fresh sync can mark venv files UF_HIDDEN; Python >= 3.12.4 then
# skips .pth files and the editable install never reaches sys.path.
if [ "$(uname)" = "Darwin" ]; then chflags -R nohidden .venv 2>/dev/null || true; fi

uv run --no-sync python - << 'EOF'
import jaxscale_lm.data.loader  # the package the unanchored ignore rule lost
import jaxscale_lm.training.trainer
from jaxscale_lm.config import load_config

config = load_config("configs/train/cpu_smoke.yaml")
assert config.model.num_layers >= 1
print(f"clean-export: imports + config validation OK "
      f"(run_name={config.project.run_name}, "
      f"{config.model.num_layers}L x {config.model.hidden_size}h)")
EOF

echo "clean-export: PASS"
