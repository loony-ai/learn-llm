"""Chapter 2: check that the environment matches what the book was tested with.

Run from `code/`:
    python -m scripts.ch02_check_env

Exits with status 0 if every check passes, 1 otherwise, so it can also be used
in automated checks.
"""

from __future__ import annotations

import importlib.metadata
import platform
import site
import sys

MIN_PYTHON = (3, 12)
EXPECTED = {"numpy": "2.5.3", "torch": "2.14.1", "pytest": "9.1.1"}


def installed_version(package: str) -> str | None:
    """Version of `package` installed in this environment's package directories, or None.

    We search only the environment's site-packages directories, not the whole import
    path, because the import path includes the current directory. An editable install
    leaves `llmfp.egg-info` in code/, which would otherwise make llmfp look installed
    for *any* Python started from code/.
    """
    directories = site.getsitepackages()
    if site.ENABLE_USER_SITE:
        directories.append(site.getusersitepackages())
    for distribution in importlib.metadata.distributions(name=package, path=directories):
        return distribution.version
    return None


def main() -> int:
    problems: list[str] = []
    print(f"Python      {platform.python_version()}  ({sys.executable})")
    if sys.version_info < MIN_PYTHON:
        problems.append(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")

    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual env {'yes' if in_venv else 'NO'}  (prefix: {sys.prefix})")
    if not in_venv:
        problems.append("not running inside a virtual environment")

    for package, expected in EXPECTED.items():
        found = installed_version(package)
        # Local labels such as "+cpu" describe the build, not the version, so ignore them.
        matches = found is not None and found.split("+")[0] == expected
        print(f"{package:<11} {found or 'missing':<14} expected {expected}  {'ok' if matches else 'MISMATCH'}")
        if not matches:
            problems.append(f"{package}: found {found}, expected {expected}")

    llmfp_version = installed_version("llmfp")
    print(f"llmfp       {llmfp_version or 'NOT INSTALLED'}")
    if llmfp_version is None:
        problems.append('llmfp is not installed; run: python -m pip install -e ".[dev]"')

    try:
        import torch

        print(f"torch CUDA  available={torch.cuda.is_available()}  build={torch.version.cuda}")
        print(f"torch MPS   available={torch.backends.mps.is_available()}")
    except ImportError:
        pass  # already reported as missing above

    if problems:
        print("\nProblems found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
