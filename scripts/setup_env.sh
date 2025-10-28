#!/usr/bin/env bash
set -euo pipefail

# config (override via env vars)
PY_VERSION="${PY_VERSION:-3.10.14}"
VENV_DIR="${VENV_DIR:-venv}"

# mandated Torch build (Linux/WSL with NVIDIA)
TORCH_VER="${TORCH_VER:-2.4.1+cu121}"
TORCH_IDX="${TORCH_IDX:-https://download.pytorch.org/whl/cu121}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# flags
FORCE_PYENV_REINSTALL=false
APT_DEPS=true      # set false to skip apt installs
HEADLESS=true      # exports MUJOCO_GL=egl

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-pyenv-reinstall) FORCE_PYENV_REINSTALL=true; shift ;;
    --no-apt) APT_DEPS=false; shift ;;
    --no-headless) HEADLESS=false; shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--force-pyenv-reinstall] [--no-apt] [--no-headless]
Env vars:
  PY_VERSION=<3.10.x>  default: $PY_VERSION
  VENV_DIR=<dir>       default: $VENV_DIR
  TORCH_VER=<ver>      default: $TORCH_VER
  TORCH_IDX=<url>      default: $TORCH_IDX
EOF
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "==> Repo: $ROOT_DIR"
echo "==> Target Python: $PY_VERSION"
echo "==> Venv dir: $VENV_DIR"

maybe_apt_install() {
  if $APT_DEPS && command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y "$@"
  fi
}

# ensure pyenv 
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"

if $FORCE_PYENV_REINSTALL && [[ -d "$PYENV_ROOT" ]]; then
  echo "==> Backing up existing $PYENV_ROOT to ${PYENV_ROOT}.bak.$(date +%s)"
  mv "$PYENV_ROOT" "${PYENV_ROOT}.bak.$(date +%s)"
fi

# try to initialize existing pyenv if present but not on PATH
if [[ -d "$PYENV_ROOT" ]]; then
  # enable shims/functions for this process even if shell integration wasn't set up
  eval "$(pyenv init -)"
fi

if ! command -v pyenv >/dev/null 2>&1; then
  echo "==> Installing pyenv and build deps…"
  maybe_apt_install build-essential curl git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
  curl https://pyenv.run | bash
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  # persist to bashrc if not already present (optional)
  if ! grep -q 'pyenv init' "$HOME/.bashrc" 2>/dev/null; then
    {
      echo ''
      echo '# >>> pyenv >>>'
      echo 'export PYENV_ROOT="$HOME/.pyenv"'
      echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
      echo 'eval "$(pyenv init -)"'
      echo '# <<< pyenv <<<'
    } >> "$HOME/.bashrc"
  fi
fi

echo "==> pyenv: $(pyenv --version)"

# install & locate python
# install if missing (idempotent with -s)
pyenv install -s "$PY_VERSION"

# get the prefix dir for that version and derive the interpreter path
PY_PREFIX="$(pyenv prefix "$PY_VERSION")"
PYBIN="${PY_PREFIX}/bin/python"
echo "==> Using interpreter: $PYBIN"
"$PYBIN" -V

# recreate venv
echo "==> Recreating $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYBIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# system libs for MuJoCo + FFmpeg
maybe_apt_install libgl1 libegl1 libosmesa6 ffmpeg || true

# python deps (clean slate)
pip install --upgrade pip
# kill any accidental old installs
pip uninstall -y gym cleanrl gymnasium || true

# 1) torch first (mandated cu121 wheel)
pip install "torch==${TORCH_VER}" --index-url "${TORCH_IDX}"

# 2) CleanRL from GitHub, but WITHOUT deps (so it won't pull legacy gym)
pip install --no-deps "cleanrl @ git+https://github.com/vwxyzjn/cleanrl.git"

# 3) rest of the stack from requirements.txt (NO cleanrl in that file)
pip install -r requirements.txt

# patch CleanRL wrappers for Gymnasium 1.x (v5 envs)
python - <<'PY'
import io, re, shutil, inspect, importlib

modules = [
    "cleanrl.ppo_continuous_action",
    "cleanrl.sac_continuous_action",
    "cleanrl.td3_continuous_action",
]

pat = re.compile(
    r"""gym\.wrappers\.TransformObservation\(
        \s*env\s*,\s*
        (?P<fn>
            (?:[^()]*\([^()]*\)[^()]*) |
            (?:[^()]+)
        )
        \s*\)
    """,
    re.VERBOSE,
)

def patch_source(src: str):
    n=0
    def repl(m):
        nonlocal n
        call = m.group(0)
        if "observation_space=" in call:
            return call
        n += 1
        fn = m.group("fn").strip()
        return f"gym.wrappers.TransformObservation(env, {fn}, observation_space=env.observation_space)"
    new = pat.sub(repl, src)
    return new, n

total=0; touched=[]
for modname in modules:
    try:
        mod = importlib.import_module(modname)
        path = inspect.getsourcefile(mod)
        if not path: 
            print(f"  - Could not locate {modname}; skipping.")
            continue
        src = io.open(path, "r", encoding="utf-8").read()
        new, n = patch_source(src)
        if n>0:
            shutil.copy2(path, path + ".bak")
            io.open(path, "w", encoding="utf-8").write(new)
            print(f"Patched {modname} ({n} change{'s' if n!=1 else ''})")
            touched.append(path)
            total += n
        else:
            print(f"No changes needed in {modname}")
    except ModuleNotFoundError:
        print(f"{modname} not installed (ok if you won't use it)")
    except Exception as e:
        print(f"Error patching {modname}: {e}")

# import check (ppo)
try:
    importlib.import_module("cleanrl.ppo_continuous_action")
    print("Import check: cleanrl.ppo_continuous_action SUCCESS")
except Exception as e:
    print("Import check: cleanrl.ppo_continuous_action FAIL", e)
print(f"Summary: {total} total changes.")
PY

# headless rendering default
if $HEADLESS; then
  export MUJOCO_GL=egl
  if ! grep -q 'MUJOCO_GL' "$VENV_DIR/bin/activate"; then
    echo 'export MUJOCO_GL=egl' >> "$VENV_DIR/bin/activate"
  fi
fi

# verify
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
except Exception as e:
    print("Torch import error:", e)
try:
    import gym
    print("Legacy gym is importable from:", gym.__file__)
except Exception as e:
    print("Legacy gym not importable:", type(e).__name__)
try:
    import gymnasium as gymn
    print("Gymnasium:", gymn.__version__)
except Exception as e:
    print("Gymnasium import error:", e)
try:
    import mujoco
    print("MuJoCo:", mujoco.__version__)
except Exception as e:
    print("MuJoCo import error:", e)
try:
    import cleanrl
    import importlib.metadata as im
    print("CleanRL path:", cleanrl.__file__)
    try:
        print("CleanRL dist version:", im.version("cleanrl"))
    except Exception:
        print("CleanRL dist version: (metadata not set, OK for GitHub install)")
except Exception as e:
    print("CleanRL import error:", e)
PY

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
