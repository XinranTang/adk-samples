"""Skill definitions for the SWE Agent."""

from google.adk.skills import models, scripts
from .. import benchmark_tools


def create_file_editor_skill(tools: benchmark_tools.BenchmarkTools) -> models.Skill:
    """Creates the file_editor skill."""
    return models.Skill(
        frontmatter=models.Frontmatter(
            name="file_editor",
            description="Utilities for reading and editing files with line numbers and search/replace logic.",
        ),
        instructions=(
            """
            # File Editor Skill

This skill provides robust tools for file manipulation, designed to handle large codebases and complex edits.

## Tool: Read File with Line Numbers
Use this to read code files. It provides line numbers which are crucial for referencing code in discussions or edits.

**Usage:**

python read_file.py <file_path> [--start_line N] [--end_line N]


**Arguments:**
- `<file_path>`: Path to the file to read.
- `--start_line N`: (Optional) Start reading from line N (1-based).
- `--end_line N`: (Optional) Stop reading at line N (1-based). Use -1 to read until end.

**Example:**

python read_file.py /testbed/src/main.py --start_line 10 --end_line 50


## Tool: Edit File
Use this to modify files using a search-and-replace block format. This format is safer than rewriting entire files and handles context validation.

**Usage:**
1. Create a temporary file containing your diff/patch (e.g. `patch.txt`).
2. Run the edit script targeting the code file and pointing to your patch file.


python edit_file.py --target_file <file_to_edit> --diff_file <path_to_patch_file>


**Diff Format:**
The `patch.txt` content MUST follow this EXACT format:
```
<<<<<<< SEARCH
[Code to replace - must match existing file exactly]
=======
[New code to insert]
>>>>>>> REPLACE
```

**Rules:**
- multiple `<<<<<<< SEARCH` blocks allowed.
- context must be unique.
- indentation must be exact.

**Example:**
```bash
# Step 1: Create patch file (using create_file tool or echo)
cat > patch.txt <<EOF
<<<<<<< SEARCH
    def foo():
        pass
=======
    def foo():
        return "bar"
>>>>>>> REPLACE
EOF

# Step 2: Apply patch
python edit_file.py --target_file /testbed/src/main.py --diff_file patch.txt


## Tool: Undo Edit
Revert a file to its state before the last `edit_file` operation.

**Usage:**

python undo_last_edit.py <target_file>

            """
        ),
        resources=models.Resources(
            scripts={
                "read_file.py": scripts.FunctionScript(tools.read_file),
                "create_file.py": scripts.FunctionScript(tools.create_file),
                "edit_file.py": scripts.FunctionScript(tools.edit_file),
                "undo_last_edit.py": scripts.FunctionScript(tools.undo_last_edit),
            }
        ),
    )


def create_shell_skill(tools: benchmark_tools.BenchmarkTools) -> models.Skill:
    """Creates the shell-commands skill."""
    return models.Skill(
        frontmatter=models.Frontmatter(
            name="shell-commands",
            description="Run any shell commands using /bin/sh. For example: 1. Grep a file and view specific lines. 2. View the list of files in the repo. 3. Run tests to see if your changes work. 4. Use piping to chain multiple commands.",
        ),
        instructions="""
# Shell Commands Skill

Capabilities for executing shell commands to explore, test, and verify the codebase.

## Tool: Run Shell Command
Executes a shell command using `/bin/sh`.

**Usage:**

python run_shell_command.py <cmd>


**Arguments:**
- `<cmd>`: The command to run.

**Examples:**

# List all files
python run_shell_command.py "find . -maxdepth 2 -not -path '*/.*'"

# Run specific tests
python run_shell_command.py "pytest tests/test_core.py"

      """,
        resources=models.Resources(
            scripts={
                "run_shell_command.py": scripts.FunctionScript(tools.run_shell_command),
            }
        )
    )
