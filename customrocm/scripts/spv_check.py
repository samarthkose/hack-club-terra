#!/usr/bin/env python3
"""spv_check.py — static SPIR-V sanity checker for CROC kernel modules.

aipc has no spirv-val, so this covers the defect class that segfaulted RADV
on 2026-10-13: pointer/result type mismatches in the load/store/access-chain
family, plus structural errors (truncated stream, unknown opcode, bad magic).

Not a full validator — a targeted guardrail. When SPIRV-Tools ever lands on
aipc, prefer spirv-val.

Usage:
    spv_check.py <file.spv | hex-dump.txt> [--quiet]

Input formats:
    * binary .spv (magic 0x07230203 little-endian)
    * text: one hex word per line (output of tools/spv_dump.c)

Exit codes: 0 = no violations, 1 = violations found, 2 = usage/IO error
"""
import sys

MAGIC = 0x07230203

# opcode -> name (subset covering CROC-emitted and glslang-emitted modules;
# unknown opcodes are reported as structural errors, which is intentional)
OPCODES = {
    1: "OpNop", 2: "OpSourceContinued", 3: "OpSource", 4: "OpSourceExtension",
    5: "OpName", 6: "OpMemberName", 7: "OpString", 8: "OpLine", 10: "OpExtension",
    11: "OpExtInstImport", 12: "OpExtInst", 14: "OpMemoryModel", 15: "OpEntryPoint",
    16: "OpExecutionMode", 17: "OpCapability", 19: "OpTypeVoid", 20: "OpTypeBool",
    21: "OpTypeInt", 22: "OpTypeFloat", 23: "OpTypeVector", 24: "OpTypeMatrix",
    25: "OpTypeImage", 26: "OpTypeSampler", 27: "OpTypeSampledImage",
    28: "OpTypeArray", 29: "OpTypeRuntimeArray", 30: "OpTypeStruct",
    31: "OpTypeOpaque", 32: "OpTypePointer", 33: "OpTypeFunction",
    41: "OpConstantTrue", 42: "OpConstantFalse", 43: "OpConstant",
    44: "OpConstantComposite", 45: "OpConstantSampler", 46: "OpConstantNull",
    54: "OpFunction", 55: "OpFunctionParameter", 56: "OpFunctionEnd",
    57: "OpFunctionCall", 59: "OpVariable", 60: "OpImageTexelPointer",
    61: "OpLoad", 62: "OpStore", 63: "OpCopyMemory", 64: "OpCopyMemorySized",
    65: "OpAccessChain", 66: "OpInBoundsAccessChain", 67: "OpPtrAccessChain",
    71: "OpDecorate", 72: "OpMemberDecorate", 73: "OpDecorationGroup",
    74: "OpGroupDecorate", 75: "OpGroupMemberDecorate",
    79: "OpVectorShuffle", 80: "OpCompositeConstruct", 81: "OpCompositeExtract",
    82: "OpCompositeInsert", 83: "OpCopyObject", 84: "OpTranspose",
    109: "OpConvertFToU", 110: "OpConvertFToS", 124: "OpBitcast",
    126: "OpSNegate", 127: "OpFNegate", 128: "OpIAdd", 129: "OpFAdd",
    130: "OpISub", 131: "OpFSub", 132: "OpIMul", 133: "OpFMul",
    134: "OpUDiv", 135: "OpSDiv", 136: "OpFDiv", 137: "OpUMod",
    138: "OpSRem", 139: "OpSMod",
    164: "OpLogicalEqual", 165: "OpLogicalNotEqual", 166: "OpLogicalOr",
    167: "OpLogicalAnd", 168: "OpLogicalNot", 169: "OpSelect",
    170: "OpIEqual", 171: "OpINotEqual", 172: "OpUGreaterThan",
    173: "OpSGreaterThan", 174: "OpUGreaterThanEqual", 175: "OpSGreaterThanEqual",
    176: "OpULessThan", 177: "OpSLessThan", 178: "OpULessThanEqual",
    179: "OpSLessThanEqual", 180: "OpFOrdEqual",
    245: "OpPhi", 246: "OpLoopMerge", 247: "OpSelectionMerge",
    248: "OpLabel", 249: "OpBranch", 250: "OpBranchConditional",
    251: "OpSwitch", 252: "OpKill", 253: "OpReturn", 254: "OpReturnValue",
    255: "OpUnreachable",
}

# opcodes whose operand 0 is the result <id> and operand 1... vary
VALUE_OPS = {
    61, 65, 66, 67, 81, 80, 79, 83, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 177, 178, 179, 245, 57, 12,
}

TYPE_BUILDERS = {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}


def parse_words(path):
    data = open(path, "rb").read()
    if data[:4] == b"\x03\x02\x23\x07":
        if len(data) % 4:
            return None, ["file size not a multiple of 4"]
        return [int.from_bytes(data[i:i + 4], "little")
                for i in range(0, len(data), 4)], []
    # text hex dump
    words = []
    errs = []
    for ln, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            words.append(int(line, 16))
        except ValueError:
            errs.append(f"text line {ln}: not a hex word: {line!r}")
    return words, errs


def decode(words, errors):
    """Structural pass: header + instruction stream alignment."""
    if len(words) < 5:
        errors.append("module shorter than 5-word header")
        return []
    if words[0] != MAGIC:
        errors.append(f"bad magic: {words[0]:08x}")
        return []
    bound = words[3]
    insns = []
    i = 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if wc == 0:
            errors.append(f"word {i}: zero word count (stream misaligned)")
            break
        if i + wc > len(words):
            errors.append(f"word {i}: instruction overruns end of module")
            break
        name = OPCODES.get(op)
        if name is None:
            errors.append(f"word {i}: unknown opcode {op} (stream misaligned?)")
            break
        insns.append((i, name, op, words[i + 1:i + wc]))
        i += wc
    if i != len(words) and not errors:
        errors.append(f"stream ended at {i}/{len(words)} words")
    if bound == 0:
        errors.append("id bound is zero")
    return insns


def check(words):
    """Type-consistency pass. Returns list of violation strings."""
    errors = []
    insns = decode(words, errors)
    if errors:
        return errors

    types = {}     # id -> ('void',) ('bool',) ('int',bits,signed) ('float',bits)
                   #       ('vector',comp,count) ('struct',[members])
                   #       ('array',elem) ('runtime_array',elem)
                   #       ('ptr',sc,pointee) ('fn_t',ret,params)
                   #       ('fn',ret)
    consts = {}    # id -> (type_id, value)
    vars_ = {}     # id -> type_id (OpVariable result)
    valtype = {}   # id -> type_id (instruction results)
    labels = set()

    # ---- pass 1: collect definitions ----
    # minimal operand counts (guards against malformed instructions; this is
    # how the missing OpTypeVector component-count bug gets caught)
    TYPE_MIN = {19: 1, 20: 1, 21: 3, 22: 2, 23: 3, 24: 3, 25: 5, 26: 4, 27: 3,
                28: 2, 29: 2, 30: 2, 31: 2, 32: 3, 33: 2}
    OP_MIN = {43: 3, 54: 4, 59: 3, 61: 3, 62: 2, 65: 3, 66: 3, 81: 4,
              245: 4, 246: 3, 247: 3, 248: 1, 250: 3, 176: 4, 128: 4}
    for pos, name, op, ops in insns:
        if op in TYPE_BUILDERS and ops:
            tid = ops[0]
            need = TYPE_MIN.get(op, 1)
            if len(ops) < need:
                errors.append(f"word {pos} {name}: expected >= {need} operands, "
                              f"got {len(ops)} (malformed type instruction)")
                continue
            if op == 19:   types[tid] = ("void",)
            elif op == 20: types[tid] = ("bool",)
            elif op == 21: types[tid] = ("int", ops[1], ops[2])
            elif op == 22: types[tid] = ("float", ops[1])
            elif op == 23: types[tid] = ("vector", ops[1], ops[2])
            elif op == 24: types[tid] = ("matrix", ops[1], ops[2])
            elif op == 28: types[tid] = ("array", ops[1])
            elif op == 29: types[tid] = ("runtime_array", ops[1])
            elif op == 30: types[tid] = ("struct", list(ops[1:]))
            elif op == 32: types[tid] = ("ptr", ops[1], ops[2])
            elif op == 33: types[tid] = ("fn_t", ops[1], list(ops[2:]))
            else:          types[tid] = ("opaque", op)
        elif op == 43 and len(ops) >= 2:          # OpConstant
            consts[ops[1]] = (ops[0], ops[2])
        elif op in (41, 42, 44, 45, 46) and len(ops) >= 2:
            consts[ops[1]] = (ops[0], None)
        elif op == 59 and len(ops) >= 2:          # OpVariable
            vars_[ops[1]] = ops[0]
            valtype[ops[1]] = ops[0]
        elif op == 54 and len(ops) >= 2:          # OpFunction
            valtype[ops[1]] = ops[0]
        elif op == 248 and ops:                   # OpLabel
            labels.add(ops[0])
        elif op in VALUE_OPS and len(ops) >= 2:
            valtype[ops[1]] = ops[0]

    def type_of(iid):
        if iid in valtype: return valtype[iid]
        if iid in consts:  return consts[iid][0]
        return None

    # Alias every value/variable result id to its declared type's descriptor,
    # so pointer-typed ids (variables, access-chain results, loads) resolve.
    for vid, tid in list(valtype.items()):
        if tid in types and vid not in types:
            types[vid] = types[tid]

    def tname(tid):
        t = types.get(tid)
        return f"%{tid}{'' if t is None else ' ' + str(t)}"

    # ---- pass 2: type checks ----
    for pos, name, op, ops in insns:
        try:
            if op in OP_MIN and len(ops) < OP_MIN[op]:
                errors.append(f"word {pos} {name}: expected >= {OP_MIN[op]} operands, "
                              f"got {len(ops)}")
                continue
            if op == 61 and len(ops) == 3:                       # OpLoad
                rt, rid, ptr = ops
                pt = types.get(ptr)
                if pt is None or pt[0] != "ptr":
                    errors.append(f"word {pos} OpLoad %{rid}: operand %{ptr} is not a pointer ({tname(ptr)})")
                elif pt[2] != rt:
                    errors.append(f"word {pos} OpLoad %{rid}: result type {tname(rt)} "
                                  f"!= pointee of %{ptr} which is {tname(pt[2])}")
            elif op == 62 and len(ops) == 2:                     # OpStore
                ptr, obj = ops
                pt = types.get(ptr)
                if pt is None or pt[0] != "ptr":
                    errors.append(f"word {pos} OpStore: operand %{ptr} is not a pointer ({tname(ptr)})")
                else:
                    ot = type_of(obj)
                    if ot is None:
                        errors.append(f"word {pos} OpStore: object %{obj} has unknown type")
                    elif ot != pt[2]:
                        errors.append(f"word {pos} OpStore: object %{obj} type {tname(ot)} "
                                      f"!= pointee {tname(pt[2])} of %{ptr}")
            elif op in (65, 66) and len(ops) >= 3:               # OpAccessChain
                rt, rid, base = ops[0], ops[1], ops[2]
                idxs = ops[3:]
                bt = types.get(base)
                if bt is None or bt[0] != "ptr":
                    errors.append(f"word {pos} OpAccessChain %{rid}: base %{base} "
                                  f"is not a pointer ({tname(base)})")
                    continue
                sc, cur = bt[1], bt[2]
                ok = True
                for ix in idxs:
                    ct = types.get(cur)
                    if ct is None:
                        errors.append(f"word {pos} OpAccessChain %{rid}: indexing non-type %{cur}")
                        ok = False; break
                    if ct[0] == "struct":
                        if ix not in consts:
                            errors.append(f"word {pos} OpAccessChain %{rid}: struct index %{ix} "
                                          f"is not a constant")
                            ok = False; break
                        member = consts[ix][1]
                        if member is None or member >= len(ct[1]):
                            errors.append(f"word {pos} OpAccessChain %{rid}: struct index out of range")
                            ok = False; break
                        cur = ct[1][member]
                    elif ct[0] in ("array", "runtime_array"):
                        cur = ct[1]
                    elif ct[0] == "vector":
                        cur = ct[1]   # component type (ct[2] is the COUNT)
                    elif ct[0] == "matrix":
                        cur = ct[1]
                    else:
                        errors.append(f"word {pos} OpAccessChain %{rid}: cannot index into {tname(cur)}")
                        ok = False; break
                if not ok:
                    continue
                rt_t = types.get(rt)
                if rt_t is None or rt_t[0] != "ptr":
                    errors.append(f"word {pos} OpAccessChain %{rid}: result type {tname(rt)} "
                                  f"is not a pointer")
                elif rt_t[1] != sc or rt_t[2] != cur:
                    errors.append(f"word {pos} OpAccessChain %{rid}: result type {tname(rt)} "
                                  f"should be ptr(sc={sc}, pointee={tname(cur)}) — "
                                  f"result type must point at the FINAL indexed type")
            elif op == 245 and len(ops) >= 3:                    # OpPhi
                rt = ops[0]
                pairs = ops[2:]
                if len(pairs) % 2:
                    errors.append(f"word {pos} OpPhi: odd operand count")
                    continue
                for k in range(0, len(pairs), 2):
                    val, blk = pairs[k], pairs[k + 1]
                    vt = type_of(val)
                    if vt is None:
                        errors.append(f"word {pos} OpPhi: incoming value %{val} unknown type")
                    elif vt != rt:
                        errors.append(f"word {pos} OpPhi: incoming %{val} type {tname(vt)} "
                                      f"!= result type {tname(rt)}")
                    if blk not in labels:
                        errors.append(f"word {pos} OpPhi: parent %{blk} is not a label")
            elif op in (176, 177, 178, 179, 170, 171) and len(ops) == 4:
                rt, rid, a, b = ops
                if types.get(rt) != ("bool",):
                    errors.append(f"word {pos} {name} %{rid}: result type {tname(rt)} is not bool")
                ta, tb = type_of(a), type_of(b)
                if ta is None or tb is None or ta != tb:
                    errors.append(f"word {pos} {name} %{rid}: operand types differ "
                                  f"({tname(ta)} vs {tname(tb)})")
            elif op in (128, 130, 132, 134, 137) and len(ops) == 4:
                rt, rid, a, b = ops
                ta, tb = type_of(a), type_of(b)
                if ta is None or tb is None or ta != rt or tb != rt:
                    errors.append(f"word {pos} {name} %{rid}: operand/result types mismatch "
                                  f"(result {tname(rt)}, operands {tname(ta)}, {tname(tb)})")
            elif op == 81 and len(ops) >= 4:                     # OpCompositeExtract
                rt, rid, comp = ops[0], ops[1], ops[2]
                cur = type_of(comp)
                ok = cur is not None
                for lit in ops[3:]:
                    if not ok:
                        break
                    c = types.get(cur)
                    if c is None:
                        ok = False; break
                    if c[0] in ("vector", "matrix", "array", "runtime_array"):
                        cur = c[1]
                    elif c[0] == "struct":
                        if lit >= len(c[1]):
                            ok = False; break
                        cur = c[1][lit]
                    else:
                        ok = False; break
                if not ok:
                    errors.append(f"word {pos} OpCompositeExtract %{rid}: cannot "
                                  f"extract from {tname(type_of(comp))}")
                elif cur != rt:
                    errors.append(f"word {pos} OpCompositeExtract %{rid}: result type {tname(rt)} "
                                  f"!= extracted type {tname(cur)}")
            elif op == 59 and len(ops) == 3:                     # OpVariable
                pt, rid, sc = ops
                t = types.get(pt)
                if t is None or t[0] != "ptr":
                    errors.append(f"word {pos} OpVariable %{rid}: result type {tname(pt)} is not a pointer")
                elif t[1] != sc:
                    errors.append(f"word {pos} OpVariable %{rid}: pointer storage class {t[1]} "
                                  f"!= variable storage class {sc}")
        except (IndexError, KeyError) as e:
            errors.append(f"word {pos} {name}: malformed operands ({e})")

    return errors


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        words, errs = parse_words(args[0])
    except OSError as e:
        print(f"cannot read {args[0]}: {e}", file=sys.stderr)
        return 2
    if words is None:
        errs = errs or ["unparseable input"]
    if errs:
        errors = list(errs)
    else:
        try:
            errors = check(words)
        except Exception as e:   # never crash — report as a violation
            errors = [f"internal checker error: {type(e).__name__}: {e}"]
    if errors:
        print(f"FAIL: {len(errors)} violation(s) in {args[0]}")
        for e in errors:
            print(f"  - {e}")
        return 1
    if not quiet:
        print(f"PASS: {args[0]} ({len(words)} words, type checks clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))