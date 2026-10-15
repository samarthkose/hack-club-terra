#!/usr/bin/env python3
"""Convert a SPIR-V binary into a C header with a uint32_t array.

Usage: spv_to_header.py <input.spv> <array_name> [guard_macro]

Prints the header to stdout.
"""
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    spv_path, name = sys.argv[1], sys.argv[2]
    guard = sys.argv[3] if len(sys.argv) > 3 else None

    data = open(spv_path, "rb").read()
    if len(data) % 4:
        print(f"{spv_path}: size {len(data)} not a multiple of 4", file=sys.stderr)
        return 1
    if data[:4] != b"\x03\x02\x23\x07":
        print(f"{spv_path}: bad SPIR-V magic", file=sys.stderr)
        return 1

    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]

    if guard:
        print(f"#ifndef {guard}")
        print(f"#define {guard}")
    print(f"/* generated from {spv_path} — do not edit */")
    print(f"static const unsigned int {name}[{len(words)}] = {{")
    for i in range(0, len(words), 6):
        row = ", ".join(f"0x{w:08x}u" for w in words[i:i + 6])
        print(f"    {row},")
    print("};")
    if guard:
        print("#endif")
    return 0


if __name__ == "__main__":
    sys.exit(main())