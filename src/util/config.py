"""Config loading. A pending measurement must fail loudly, not default."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


class PendingMeasurement(RuntimeError):
    """Raised when code needs a value that has not been measured yet."""


class Thresholds:
    def __init__(self, d): self._d = d
    def __getattr__(self, name):
        if name not in self._d:
            raise AttributeError(name)
        v = self._d[name]
        if v is None:
            raise PendingMeasurement(
                f"'{name}' is not measured yet. See configs/thresholds.yaml "
                f"for which measurement it is waiting on.")
        return v
    def is_pending(self, name): return self._d.get(name) is None
    def pending(self): return [k for k, v in self._d.items() if v is None]


def load_thresholds(path=None):
    path = path or ROOT / "configs" / "thresholds.yaml"
    return Thresholds(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_base(path=None):
    path = path or ROOT / "configs" / "base.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
