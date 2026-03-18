"""
Minimal C preprocessor for OpenCUDA.

Handles:
  #define NAME VALUE  — text substitution
  #define NAME        — define without value (for #ifdef)
  // and /* */ comments — already handled by lexer
"""

from __future__ import annotations
import re


def preprocess(source: str) -> str:
    """Apply #define substitutions to source code."""
    defines: dict[str, str] = {}
    output_lines = []

    for line in source.split('\n'):
        stripped = line.strip()

        # #define NAME VALUE
        m = re.match(r'#define\s+(\w+)\s+(.*)', stripped)
        if m:
            name, value = m.group(1), m.group(2).strip()
            defines[name] = value
            output_lines.append('')  # preserve line numbers
            continue

        # #define NAME (no value)
        m = re.match(r'#define\s+(\w+)\s*$', stripped)
        if m:
            defines[m.group(1)] = '1'
            output_lines.append('')
            continue

        # #include — skip (not supported)
        if stripped.startswith('#include'):
            output_lines.append('')
            continue

        # #pragma — skip
        if stripped.startswith('#pragma'):
            output_lines.append('')
            continue

        # Skip other preprocessor directives
        if stripped.startswith('#'):
            output_lines.append('')
            continue

        # Apply substitutions (whole-word only)
        result = line
        for name, value in sorted(defines.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(name) + r'\b', value, result)

        output_lines.append(result)

    return '\n'.join(output_lines)
