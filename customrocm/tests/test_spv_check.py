#!/usr/bin/env python3
"""Tests for scripts/spv_check.py.

test_spv_check

The 'buggy' module is reconstructed word-for-word from the real spv_dump
output of the hand-emitted fill kernel BEFORE the 2026-10-13 fix (the module
that segfaulted RADV). The 'fixed' variant differs only in the three
OpAccessChain result types (pointer-to-struct -> pointer-to-uint).

Expected:
buggy  -> FAIL with >=3 OpAccessChain result-type violations
fixed  -> PASS
truncated stream -> structural FAIL
unknown opcode   -> structural FAIL

Run:  python3 tests/test_spv_check.py
"""
