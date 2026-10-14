# CROC: Custom ROCm On Compute (Vulkan backend)

A Vulkan compute runtime for 4x AMD RX 580 (gfx803) cards in the `aipc` box.
It sidesteps the KFD PCIe-atomics check that leaves 2 of the 4 GPUs unusable
by driving everything through Vulkan compute queues instead of ROCm/HSA.

## Why bother

`aipc` has four RX 580s in it. Stock ROCm/KFD only sees two of them, because
the Z390 PCH root ports don't advertise PCIe AtomicOps capability and KFD
refuses to touch anything behind them. Vulkan through RADV doesn't care. It
enumerates all four cards and runs kernels on them just fine.

## Build

The main build path on `aipc` is plain `make`. That machine has gcc and make,
no cmake, no sudo, and I'd like to keep it that way.

```bash
# From Windows (Git Bash), once aipc is powered on:
tools/transfer.sh          # tar-over-ssh + make + crocinfo + both demo paths + checks

# Or by hand on aipc:
make -j4
make test                  # python3 tests/test_spv_check.py (SPIR-V checker)
./build/crocinfo           # enumerate devices (tier/link/clocks)
./build/croc_layer_demo    # builtin glslang kernel, verifies results
./build/croc_layer_demo --emit   # same thing, but through the hand-rolled emitter
./build/spv_dump | python3 scripts/spv_check.py /dev/stdin   # static SPIR-V check
```

CMake still works on machines that have cmake and system Vulkan headers:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
```

**Dependencies**: the Vulkan loader (`libvulkan.so.1`, ships with Mesa) and a
driver that speaks Vulkan (RADV). No ROCm packages, no kernel rebuild, no sudo.

**Toolchain on aipc (all user-local, no sudo):**
- `~/sdk/Vulkan-Headers` — headers to compile against, since there's no
  `libvulkan-dev` installed
- `~/sdk/glslang` — glslangValidator 16.6.0, compiles `shaders/*.comp` to
  SPIR-V at build time and `scripts/spv_to_header.py` embeds it into
  `build/fill_spv.h`
- Makefile links `-l:libvulkan.so.1` because there's no dev symlink around

## Moving it to aipc

```bash
# From Windows (Git Bash / WSL), once aipc is up:
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

// One buffer per device, dispatch the kernel on each
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

**V1 verified on aipc (2026-10-13). Real GPU compute on all 4 RX 580s.**

```
Devices: 4
Dispatching fill kernel across 4 devices...
  Device 0: 65536 words, 65536/65536 == 0xA5A50000 -> PASS   (CPU-attached x8@gen3)
  Device 1: 65536 words, 65536/65536 == 0xA5A50001 -> PASS   (CPU-attached x8@gen3)
  Device 2: 65536 words, 65536/65536 == 0xA5A50002 -> PASS   (PCH x4@gen3)
  Device 3: 65536 words, 65536/65536 == 0xA5A50003 -> PASS   (PCH x1@gen3)
Result: 4/4 devices verified
```

That includes GPUs 2 and 3 sitting behind the Z390 PCH, the two cards KFD
rejects with `PCI rejects atomics 730<0`. CROC runs verified kernels on them
through Vulkan compute.

**Emitter crash: root-caused, fixed, and verified on aipc (2026-10-13).**
The hand-rolled SPIR-V emitter in `src/kernelc.c` was segfaulting RADV inside
`vkCreateComputePipelines`. Three things were wrong:

1. `OpTypeVector` was emitted with 2 operands instead of 3 (the component
   count was missing)
2. push-constant `OpAccessChain` had the wrong result type: pointer-to-struct
   where it needed pointer-to-uint, since it has to point at the final indexed
   type
3. residual mess: the loop cluster (header-relative conditional, `OpPhi`
   induction, StorageBuffer SC, SPIR-V 1.3). Fixed by cloning the shape
   glslang already produces: SPIR-V 1.0, Uniform+BufferBlock, function-scope
   induction variable, unconditional header branch, condition parked in its
   own block

After that, `croc_layer_demo --emit` gets **4/4 devices PASS**, same as the
builtin glslang kernel. It's guarded by `scripts/spv_check.py` plus
`tests/test_spv_check.py` (the fixtures include the actual pre-fix module).
Worth noting the live-module check in `tools/transfer.sh` is what caught an
incomplete fix that the unit fixtures missed. More detail in ACTIVITY_LOG.md
sessions 4–5.

Next steps:

1. Diff the hand-emitted module against glslang output (needs
   spirv-val/spirv-dis), then fix or drop the emitter
2. More kernels: vector add, GEMM tile, L2-norm, GLSL to builtin SPIR-V
3. Multi-device peer copy and a weight-sharded matmul demo
4. llama.cpp Vulkan cross-check (`-DGGML_VULKAN=ON`,
   `GGML_VK_VISIBLE_DEVICES`)
5. Optional KFD patch lane (`kernel/`) if PyTorch-on-4 ever becomes mandatory