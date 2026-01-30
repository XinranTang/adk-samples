---
name: file_editor
description: Utilities for reading and editing files with line numbers and search/replace logic.
---

# File Editor Skill

This skill provides robust tools for file manipulation, designed to handle large codebases and complex edits.

## Tool: Read File with Line Numbers
Use this to read code files. It provides line numbers which are crucial for referencing code in discussions or edits.

**Usage:**
```bash
python3 /skills/file_editor/read_file.py <file_path> [--start_line N] [--end_line N]
```

**Arguments:**
- `<file_path>`: Path to the file to read.
- `--start_line N`: (Optional) Start reading from line N (1-based).
- `--end_line N`: (Optional) Stop reading at line N (1-based). Use -1 to read until end.

**Example:**
```bash
python3 /skills/file_editor/read_file.py /testbed/src/main.py --start_line 10 --end_line 50
```

## Tool: Edit File
Use this to modify files using a search-and-replace block format. This format is safer than rewriting entire files and handles context validation.

**Usage:**
1. Create a temporary file containing your diff/patch (e.g. `patch.txt`).
2. Run the edit script targeting the code file and pointing to your patch file.

```bash
python3 /skills/file_editor/edit_file.py --target_file <file_to_edit> --diff_file <path_to_patch_file>
```

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
python3 /skills/file_editor/edit_file.py --target_file /testbed/src/main.py --diff_file patch.txt
```

## Tool: Undo Edit
Revert a file to its state before the last `edit_file` operation.

**Usage:**
```bash
python3 /skills/file_editor/undo_edit.py <target_file>
```
