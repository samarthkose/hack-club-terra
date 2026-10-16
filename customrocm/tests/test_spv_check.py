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
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "..", "scripts", "spv_check.py")

MAGIC = 0x07230203
VERSION = 0x00010003
BOUND = 38

# (opcode, operands) — mirrors src/kernelc.c emit_fill(), taken from the real
# decoded dump of the pre-fix module. Three OpAccessChain ops are parameterized.
def instructions(fixed):
    """Module instructions. fixed=False reproduces the REAL pre-fix dump
    (both known defects present): OpTypeVector missing its component-count
    operand, and the three push-constant OpAccessChain instructions using a
    pointer-to-struct (11) result type instead of pointer-to-uint (16).
    fixed=True is the corrected module matching the fixed kernelc.c.
    Note that these values below are proably only for the rx580 GPU.
    These values can be found using AMD tools in Linux and from AMD tools
    database. Take these with a grain of salt since they might be different
    on your device. For example, these values are pulled straight from MY
    config, not YOURS."""
    vec_op = (23, [4, 2, 3]) if fixed else (23, [4, 2])
    access_t = 16 if fixed else 11
    return [
        (17, [1]),                                        # OpCapability Shader
        (14, [0, 1]),                                     # OpMemoryModel
        (15, [5, 20, 0x6E69616D, 0, 14]),                 # OpEntryPoint "main"
        (16, [20, 17, 256, 1, 1]),                        # LocalSize 256,1,1
        (71, [14, 11, 28]),                               # %gid BuiltIn 28
        (71, [9, 34, 0]), (71, [9, 33, 0]),               # ssbo set/binding
        (71, [7, 2]), (72, [7, 0, 35, 0]),                # ssbo Block, Offset
        (71, [6, 6, 4]),                                  # ArrayStride 4
        (71, [10, 2]),                                    # pc Block
        (72, [10, 0, 35, 0]), (72, [10, 1, 35, 4]), (72, [10, 2, 35, 8]),
        (19, [1]),                                        # void
        (21, [2, 32, 0]),                                 # uint
        (20, [3]),                                        # bool
        vec_op,                                           # v3uint (bug when not fixed)
        (33, [5, 1]),                                     # fn(void)
        (29, [6, 2]),                                     # runtime array uint
        (30, [7, 6]),                                     # ssbo struct
        (32, [8, 12, 7]),                                 # ptr SSBO struct
        (59, [8, 9, 12]),                                 # ssbo var
        (30, [10, 2, 2, 2]),                              # pc struct
        (32, [11, 9, 10]),                                # ptr PC struct
        (59, [11, 12, 9]),                                # pc var
        (32, [13, 1, 4]),                                 # ptr Input v3
        (59, [13, 14, 1]),                                # gid var
        (32, [15, 12, 2]),                                # ptr SSBO uint
        (32, [16, 9, 2]),                                 # ptr PC uint
        (43, [2, 17, 0]), (43, [2, 18, 1]), (43, [2, 19, 2]),
        (54, [1, 20, 0, 5]),                              # OpFunction main
        (248, [21]),                                      # entry label
        (61, [4, 26, 14]),                                # load gid
        (81, [2, 27, 26, 0]),                             # gid.x
        (65, [access_t, 28, 12, 17]),                     # &pc.count
        (61, [2, 29, 28]),
        (65, [access_t, 30, 12, 18]),                     # &pc.pattern
        (61, [2, 31, 30]),
        (65, [access_t, 32, 12, 19]),                     # &pc.stride
        (61, [2, 33, 32]),
        (249, [22]),                                      # branch loop
        (248, [22]),                                      # loop label
        (245, [2, 35, 27, 21, 37, 24]),                   # phi
        (176, [3, 34, 35, 29]),                           # ival < count
        (246, [25, 24, 0]),                               # loop merge
        (250, [34, 23, 25]),
        (248, [23]),                                      # body
        (65, [15, 36, 9, 17, 35]),                        # &words[ival]
        (62, [36, 31]),                                   # store pattern
        (249, [24]),
        (248, [24]),                                      # continue
        (128, [2, 37, 35, 33]),                           # ival + stride
        (249, [22]),
        (248, [25]),                                      # merge
        (253, []),                                        # return
        (56, []),                                         # function end
    ]


def assemble(insns):
    words = [MAGIC, VERSION, 0, BOUND, 0]
    for op, ops in insns:
        words.append(((len(ops) + 1) << 16) | op)
        words.extend(ops)
    return words


def run_checker(words, tmpdir, tag):
    path = os.path.join(tmpdir, f"{tag}.txt")
    with open(path, "w") as f:
        for w in words:
            f.write(f"{w:08x}\n")
    proc = subprocess.run([sys.executable, CHECKER, path],
                           capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        # 1. buggy module (pre-fix dump: bad vector type + bad access chains)
        rc, out = run_checker(assemble(instructions(False)), tmp, "buggy")
        if rc != 1:
            failures.append(f"buggy module: expected exit 1, got {rc}\n{out}")
        else:
            if out.count("OpAccessChain") < 3:
                failures.append(f"buggy module: expected >=3 AccessChain violations, "
                                f"got {out.count('OpAccessChain')}\n{out}")
            if "OpTypeVector" not in out:
                failures.append(f"buggy module: expected OpTypeVector operand-count "
                                f"violation\n{out}")

        # 2. fixed module
        rc, out = run_checker(assemble(instructions(True)), tmp, "fixed")
        if rc != 0:
            failures.append(f"fixed module: expected exit 0, got {rc}\n{out}")

        # 3. truncated stream
        words = assemble(instructions(True))
        rc, out = run_checker(words[:-3], tmp, "truncated")
        if rc != 1:
            failures.append(f"truncated: expected exit 1, got {rc}\n{out}")

        # 4. unknown opcode (corrupt an instruction's opcode field)
        words = assemble(instructions(True))
        words[5] = (words[5] & 0xFFFF0000) | 0xDEAD
        rc, out = run_checker(words, tmp, "badop")
        if rc != 1 or "unknown opcode" not in out:
            failures.append(f"bad opcode: expected structural FAIL, got rc={rc}\n{out}")

        # 5. OpLoad with mismatched pointee (second bug class)
        insns = instructions(True)
        for i, (op, ops) in enumerate(insns):
            if op == 61 and ops == [2, 29, 28]:
                insns[i] = (61, [3, 29, 28])   # claim bool result from uint pointer
                break
        rc, out = run_checker(assemble(insns), tmp, "badload")
        if rc != 1 or "OpLoad" not in out:
            failures.append(f"bad load: expected OpLoad violation, got rc={rc}\n{out}")

        # 6. access chain into a vector component (regression: the checker
        #    once walked the vector COUNT instead of the component type,
        #    producing a false positive on glslang's fill.spv)
        vec_mod = [
            (17, [1]), (14, [0, 1]),
            (15, [5, 10, 0x6E69616D, 0, 7]), (16, [10, 17, 1, 1, 1]),
            (71, [7, 11, 28]),
            (19, [1]), (21, [2, 32, 0]), (20, [3]), (23, [4, 2, 3]), (33, [5, 1]),
            (32, [6, 1, 4]), (59, [6, 7, 1]),        # ptr(Input, v3) gid var
            (32, [8, 1, 2]), (43, [2, 9, 0]),        # ptr(Input, uint), const 0
            (54, [1, 10, 0, 5]), (248, [11]),
            (65, [8, 12, 7, 9]),                     # chain into vector component
            (61, [2, 13, 12]),
            (253, []), (56, []),
        ]
        rc, out = run_checker(assemble(vec_mod), tmp, "vecchain")
        if rc != 0:
            failures.append(f"vector access chain: expected exit 0, got {rc}\n{out}")

        # 7. same chain with wrong result type (claims pointer-to-vector)
        bad = list(vec_mod)
        for i, (op, ops) in enumerate(bad):
            if op == 65:
                bad[i] = (65, [6, 12, 7, 9])
                break
        rc, out = run_checker(assemble(bad), tmp, "vecchain_bad")
        if rc != 1 or "OpAccessChain" not in out:
            failures.append(f"bad vector chain: expected OpAccessChain violation, "
                            f"rc={rc}\n{out}")

    if failures:
        print("TEST FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("all spv_check tests passed "
          "(buggy=FAIL with OpTypeVector + 3 AccessChain errors, fixed=PASS, "
          "truncated/bad-opcode/bad-load=FAIL, vector-chain PASS/bad=FAIL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())