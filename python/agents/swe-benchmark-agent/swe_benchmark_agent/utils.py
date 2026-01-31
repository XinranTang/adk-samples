"""Utility functions and constants for the SWE Agent."""

import logging

logger = logging.getLogger(__name__)

TIMEOUT_EXIT_CODE = 124
MEMORY_LIMIT_EXIT_CODE = 137
MAX_VERIFICATION_TURN_COUNT = 40
MAX_STATE_SIZE_BYTES = 1024


def maybe_truncate_output(
    output: str,
    max_lines: int = 50,
    max_characters_per_line: int = 320,
) -> tuple[str, bool]:
    """Truncate the output by omitting the middle lines."""
    if max_lines <= 0:
        return output, False

    lines = output.splitlines(keepends=True)
    truncated_lines = []
    for line in lines:
        if len(line) > max_characters_per_line:
             num_truncated_chars = len(line) - max_characters_per_line
             ellipsis = f"(...line too long, truncated {num_truncated_chars} characters...)"
             len_prefix = (max_characters_per_line + 1) // 2
             len_suffix = max_characters_per_line // 2
             prefix = line[:len_prefix]
             suffix = line[len(line) - len_suffix :]
             truncated_lines.append(f"{prefix}{ellipsis}{suffix}")
        else:
            truncated_lines.append(line)
            
    lines = truncated_lines
    num_lines = len(lines)

    if num_lines <= max_lines:
        return "".join(lines), False

    half_max_lines = max_lines // 2
    omitted_lines = num_lines - max_lines

    truncated_output = (
        f"(Output too large with {num_lines} lines. Only show the first and last"
        f" {half_max_lines} lines)\n"
        + "".join(lines[:half_max_lines])
        + f"\n(... truncated {omitted_lines} lines ...)\n"
        + "".join(lines[-(max_lines - half_max_lines) :])
    )

    return truncated_output, True
