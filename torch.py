import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _prepend_env_flag(name: str, flag: str) -> None:
    current = os.environ.get(name, "").strip()
    if flag in current:
        return
    os.environ[name] = f"{flag} {current}".strip()


cpu_count = os.cpu_count() or 8
os.environ.setdefault("MAX_JOBS", str(min(cpu_count, 8)))
_prepend_env_flag("NVCC_PREPEND_FLAGS", "--threads 8")

current_dir = Path(__file__).resolve().parent
search_path = []
for entry in sys.path:
    if not entry:
        continue
    try:
        if Path(entry).resolve() == current_dir:
            continue
    except OSError:
        pass
    search_path.append(entry)

spec = importlib.machinery.PathFinder.find_spec("torch", search_path)
if spec is None or spec.loader is None:
    raise ImportError("Unable to locate the real torch package")

module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)
