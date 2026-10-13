# CROC — Custom ROCm On Compute (Vulkan backend)

A Vulkan-compute-based GPU compute runtime for 4× AMD RX 580 (gfx803) on the `aipc` host. Bypasses the KFD PCIe-atomics gate that blocks 2 of 4 GPUs by using Vulkan compute queues instead of ROCm/HSA.

## Why this exists

The aipc host has 4× RX 580. Standard ROCm/KFD only exposes 2 GPU nodes because the Z390 PCH root ports don't advertise PCIe AtomicOps capability. CROC uses Vulkan compute (RADV driver) which enumerates all 4 GPUs regardless of PCIe atomics support.

## Build

Primary build path on aipc uses plain `make` (aipc has gcc + make, no cmake,
no sudo):

```bash
# From Windows (Git Bash), after aipc is powered on:
tools/transfer.sh          # tar-over-ssh + make + crocinfo + both demo paths + checks

# Or manually on aipc:
make -j4
make test                  # python3 tests/test_spv_check.py (SPIR-V checker)
./build/crocinfo           # enumerate devices (tier/link/clocks)
./build/croc_layer_demo    # builtin glslang kernel, verifies results
./build/croc_layer_demo --emit   # same, but through the hand-rolled emitter
./build/spv_dump | python3 scripts/spv_check.py /dev/stdin   # static SPIR-V check
```

CMake remains supported where cmake + system Vulkan headers exist:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
```

**Dependencies**: Vulkan loader (`libvulkan.so.1`, shipped with Mesa) and a
Vulkan-capable driver (RADV). No ROCm packages, no kernel rebuild, no sudo.

**Toolchain on aipc (user-local, no sudo):**
- `~/sdk/Vulkan-Headers` — headers for compilation (no libvulkan-dev installed)
- `~/sdk/glslang` — glslangValidator 16.6.0, compiles `shaders/*.comp` to SPIR-V
  at build time; embedded via `scripts/spv_to_header.py` into `build/fill_spv.h`
- Makefile links `-l:libvulkan.so.1` (runtime soname; no dev symlink present)

## Transfer to aipc

```bash
# From Windows (Git Bash / WSL), after aipc is powered on:
AIPC_HOST=aipc@192.168.1.144 ./tools/transfer.sh
```

## API overview

```c
#include <croc/croc.h>

// Instance + device enumeration
croc_instance* inst;
croc_instance_create(&inst);
uint32_t n = croc_instance_device_count(inst);
croc_device* dev;
croc_instance_device_get(inst, 0, &dev);

// Stream (command buffer + fence)
croc_stream* stream;
croc_stream_create(dev, &stream);

// Buffer
croc_buffer* buf;
croc_buffer_allocate(dev, 1024, &buf);
void* data;
croc_buffer_map(buf, &data);
// ... write to data ...
croc_buffer_unmap(buf);

// Kernel dispatch
croc_kernel* kern;
croc_kernel_create(dev, "main", &kern);
uint32_t wg[] = { 4, 1, 1 };
croc_kernel_dispatch(kern, wg, stream);
croc_stream_synchronize(stream);
```

## Multi-GPU layer sharding

```c
// Shard a logical layer across N devices
uint32_t shard_sizes[N];
croc_layer_plan(devices, N, total_size, shard_sizes);

// Allocate one buffer per device, dispatch kernel on each
for (i = 0; i < N; i++) {
    croc_buffer_allocate(devices[i], shard_sizes[i], &buffers[i]);
    croc_kernel_dispatch(kernels[i], wg, streams[i]);
}
for (i = 0; i < N; i++)
    croc_stream_synchronize(streams[i]);
```

## Files

| File | Purpose |
|------|---------|
| `include/croc/croc.h` | Public API |
| `include/croc/croc_internal.h` | Internal structs (Vulkan handles) |
| `src/loader.c` | Vulkan instance + physical device enumeration |
| `src/device.c` | Command buffers, dispatch, synchronization |
| `src/memory.c` | Buffer allocation, map/unmap |
| `src/kernelc.c` | SPIR-V shader compiler (minimal) |
| `src/scheduler.c` | Multi-GPU layer sharding |
| `tools/crocinfo.c` | Device enumeration tool |
| `tools/croc_layer_demo.c` | Multi-GPU layer sharding demo |
| `tools/transfer.sh` | rsync + build + test on aipc |
| `tools/gcr_atomic_probe.sh` | AtomicOpsCap diagnostic |
| `scripts/spv_check.py` | Static SPIR-V type/structure checker (aipc has no spirv-val) |
| `scripts/spv_to_header.py` | .spv → C header embedding |
| `tests/test_spv_check.py` | Checker tests, incl. the real pre-fix RADV-crashing module |
| `docs/architecture.md` | Design rationale |

## Status

**V1 VERIFIED on aipc (2026-10-13) — real GPU compute on all 4 RX 580s.**

```
Devices: 4
Dispatching fill kernel across 4 devices...
  Device 0: 65536 words, 65536/65536 == 0xA5A50000 -> PASS   (CPU-attached x8@gen3)
  Device 1: 65536 words, 65536/65536 == 0xA5A50001 -> PASS   (CPU-attached x8@gen3)
  Device 2: 65536 words, 65536/65536 == 0xA5A50002 -> PASS   (PCH x4@gen3)
  Device 3: 65536 words, 65536/65536 == 0xA5A50003 -> PASS   (PCH x1@gen3)
Result: 4/4 devices verified
```

This includes GPUs 2 and 3 behind the Z390 PCH — the cards KFD rejects with
`PCI rejects atomics 730<0`. CROC executes verified kernels on them via
Vulkan compute.

**Emitter crash — ROOT-CAUSED, FIXED, AND VERIFIED on aipc (2026-10-13).**
The hand-rolled SPIR-V emitter (`src/kernelc.c`) segfaulted RADV at
`vkCreateComputePipelines`. Three defects were found and fixed:
1. `OpTypeVector` emitted with 2 operands instead of 3 (missing component count)
2. push-constant `OpAccessChain` result type was pointer-to-struct instead of
   pointer-to-uint (must point at the FINAL indexed type)
3. residual: loop cluster (header-relative conditional + `OpPhi` induction +
   StorageBuffer SC + SPIR-V 1.3) — fixed by cloning the known-good glslang
   output shape (SPIR-V 1.0, Uniform+BufferBlock, function-scope induction
   variable, unconditional header branch, condition in a separate block)

Result: `croc_layer_demo --emit` → **4/4 devices PASS**, same as the builtin
glslang kernel. Guarded by `scripts/spv_check.py` + `tests/test_spv_check.py`
(fixtures include the real pre-fix module); the live-module check in
`tools/transfer.sh` caught an incomplete fix that unit fixtures missed.
See ACTIVITY_LOG.md sessions 4–5.

Next steps:
1. Diff hand-emitted module vs glslang output (needs spirv-val/spirv-dis);
   fix or remove the emitter
2. More kernels: vector add, GEMM tile, L2-norm — GLSL → builtin SPIR-V
3. Multi-device peer copy + weight-sharded matmul demo
4. llama.cpp Vulkan cross-check (`-DGGML_VULKAN=ON`, `GGML_VK_VISIBLE_DEVICES`)
5. Optional KFD patch lane (`kernel/`) if PyTorch-on-4 becomes mandatory