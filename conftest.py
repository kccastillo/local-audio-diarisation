"""Project-root conftest.

Adds the project root to sys.path so `from diarizer.X import Y` resolves in tests.
This is belt-and-braces against pytest 9's import-mode behaviour — the
`[tool.pytest.ini_options] pythonpath` setting in pyproject.toml should be
sufficient, but having this conftest as well makes things explicit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
