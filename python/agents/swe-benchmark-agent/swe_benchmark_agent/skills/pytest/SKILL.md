---
name: pytest
description: Run pytest on python files or directories to verify functionality.
---

# Pytest Skill

Run pytest on tests.

**Usage:**
```bash
python3 /skills/pytest/run_pytest.py <path> [options]
```

**Common Options:**
- `-v`: Verbose output.
- `--lf`: Run only tests that failed in the last run.
- `-k EXPRESSION`: Run tests matching the expression (e.g. `test_login`).

**Example:**
```bash
python3 /skills/pytest_runner/run_pytest.py /testbed/tests/test_foo.py --maxfail 1
```
