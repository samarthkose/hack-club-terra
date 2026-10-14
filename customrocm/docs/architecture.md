# CROC Architecture

CROC (Custom ROCm On Compute) is a Vulkan-compute GPU runtime for AMD RX 580
(gfx803) cards. It exists because KFD's PCIe-atomics gate blocks half the
GPUs on the aipc host, and Vulkan doesn't have that gate.

## Why Vulkan instead of ROCm/KFD

The aipc host has 4x RX 580 cards. KFD only sees 2 GPU nodes. Here's the
chain of cause and effect:

1. **Root cause**: the Z390 PCH root ports (`00:1b.0`, `00:1c.6`) don't
   advertise `AtomicOpsCap` in Device Capabilities 2
2. **The gate**: KFD logs `kfd: skipped device 1002:67df, PCI rejects
   atomics 730<0` and drops any GPU sitting behind the PCH (`03:00.0`,
   `05:00.0`)
3. **The hardware reality**: the CPU root ports (`00:01.0`, `00:01.1`) do
   support atomics, the PCH root ports don't. That's a Z390 chipset
   limitation, not a bug in any driver

The fix is to not use KFD at all. Vulkan compute queues go through RADV
(Mesa 26.0.8), which enumerates all 4 GPUs no matter what the PCIe atomics
situation looks like. CROC then builds a HIP-flavored runtime on top of
Vulkan.

## Layer structure

```
+---------------------------------------------+
|  User code (crocinfo, croc_layer_demo)      |
+---------------------------------------------+
|  croc.h public API                          |
|  - instance/device/stream/buffer/kernel     |
|  - layer_plan / layer_dispatch (multi-GPU)  |
+---------------------------------------------+
|  Internal implementation                    |
|  - loader.c: Vulkan instance + device enum  |
|  - device.c: command buffers, dispatch      |
|  - memory.c: buffer alloc/map/unmap         |
|  - kernelc.c: SPIR-V shader compiler        |
|  - scheduler.c: multi-GPU layer sharding    |
+---------------------------------------------+
|  Vulkan loader + RADV driver                |
|  (all 4 GPUs, no KFD gate)                  |
+---------------------------------------------+
```

## Key design decisions

1. **Stay away from KFD entirely.** Vulkan compute never opens `/dev/kfd`,
   so the atomics gate never comes up.
2. **Host-policed fences.** A fence signals command buffer completion and
   the CPU waits with `vkWaitForFences`. No GPU-side signal atomics
   anywhere, which sidesteps the whole reason KFD choked.
3. **Contiguous layer sharding.** `croc_layer_plan` carves a logical buffer
   into contiguous ranges, one per device. That fits weight-sharded
   inference nicely: each device owns a contiguous slice of the weight
   matrix, so you don't scatter reads across cards.
4. **Minimal dependencies.** You need the Vulkan loader and a Vulkan-capable
   driver (RADV). No ROCm packages, no kernel rebuild, no sudo.

## Device tier classification

Each device gets classified by where it sits in the PCIe topology:

- **CPU-attached (tier 0)**: GPU behind root port `00:01.0` or `00:01.1`.
  x8 link, fast.
- **PCH (tier 1)**: GPU behind PCH bridge `00:1b.0` or `00:1c.6`. x4 or x1
  link, slower for host-device transfers, but compute runs at full speed.
- **Unknown (tier 2)**: couldn't figure out the topology.

Tier changes what you should expect from H2D/D2H bandwidth. It does not
change compute throughput. All 4 GPUs are full 36 CU gfx803 parts.

## Multi-GPU sharding pattern

```c
// 1. Plan: split total_size across N devices
uint32_t shard_sizes[N];
croc_layer_plan(devices, N, total_size, shard_sizes);

// 2. Allocate: one buffer per device, sized to its shard
for (i = 0; i < N; i++)
    croc_buffer_allocate(devices[i], shard_sizes[i], &buffers[i]);

// 3. Dispatch: launch kernel on each device with its shard
for (i = 0; i < N; i++) {
    uint32_t wg[] = { (shard_sizes[i] + 255) / 256, 1, 1 };
    croc_kernel_dispatch(kernels[i], wg, streams[i]);
}

// 4. Synchronize: wait for all streams
for (i = 0; i < N; i++)
    croc_stream_synchronize(streams[i]);
```

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| PCH link x1/x4 caps H2D/D2H throughput | Keep weights resident on device; shard by layer, not tensor-page |
| 12 GB host RAM limits staging buffers | Bounded staging ring, reused across transfers |
| RADV SPIR-V compiler bugs on gfx803 | Start with minimal shaders; fall back to pre-compiled SPIR-V via `croc_kernel_compile_spirv` |
| Future kernel patches could break Vulkan | Pin the Mesa version; retest after kernel updates |

## Verification ladder

1. **crocinfo** enumerates 4 discrete RX 580s and prints tier, link width,
   and kfd_visible for each device.
2. **croc_layer_demo** shards a 1 MiB layer across all 4 devices, dispatches
   a fill kernel, synchronizes, and checks the shard sizes.
3. **llama.cpp drop-in**: build with `-DGGML_VULKAN=ON` and set
   `GGML_VK_VISIBLE_DEVICES=0,1,2,3` to force all 4 RADV devices.

## Compiler scope

**V1 (verified 2026-10-13)**: two kernel source paths coexist.

1. **Builtin GLSL kernels (default, working).** `shaders/fill.comp` gets
   compiled offline at build time by glslangValidator (user-local install,
   no sudo), embedded through `scripts/spv_to_header.py`, and loaded via
   `croc_kernel_create_spirv`. Calling `croc_kernel_create("fill")` picks
   this path first.
2. **Hand-rolled SPIR-V emitter (`kernelc.c`, fallback, still buggy).**
   Emits a strided-fill module at runtime. The module decodes fine against
   the SPIRV-Headers grammar, but it segfaults RADV during pipeline
   creation. The exact semantic defect is still unidentified since there's
   no spirv-val on aipc. It stays around as a fallback and a debugging
   exercise (`tools/spv_dump.c` dumps the module for inspection).

**Long-term (LLVM lane)**: a real LLVM AMDGPU backend targeting gfx803
would drop the Vulkan dependency entirely, but that's months of work, not
weeks. The `croc_kernel_create_spirv` seam means a native codegen backend
could slot in later without changing the public API.