# GCR: Custom ROCm on Vulkan Design Document

## 1. The Problem

The system has four RX 580 GPUs based on the gfx803 architecture. However, the standard Linux KFD driver only detects two of them.

The other two GPUs are not registered because their PCIe connections do not support the AtomicOps capability required by the driver.

The goal is to make all four GPUs available for AI workloads without modifying the Linux kernel or relying on the standard KFD compute path.

## 2. The Proposed Solution: Vulkan Compute

The proposed solution is to build a custom compute runtime on top of Vulkan.

The RADV driver can detect all four GPUs, including the two that the KFD driver cannot register. By using Vulkan compute shaders, GCR can access these GPUs without depending on the KFD atomics requirement.

### Why This Approach Should Work

The KFD atomics check, implemented through `pci_enable_atomic_req_to_root`, is related to the PCIe atomic operations required by AMD's AQL, or Asynchronous Queue, implementation.

Vulkan uses a different execution and synchronization model. Instead of relying on AMD's AQL queues, Vulkan applications use command buffers, semaphores, fences, and pipeline barriers to manage work and synchronization.

These Vulkan synchronization mechanisms do not inherently require PCIe AtomicOps support.

This gives GCR a potential way to use all four GPUs through Vulkan without changing the kernel driver.

However, successful GPU enumeration does not automatically guarantee that every compute workload will work correctly. The runtime will need to handle device memory, synchronization, shader compilation, and multi-GPU execution explicitly.

## 3. System Architecture

GCR is divided into several components. Each component handles a specific part of the compute process, from the application-facing API to GPU discovery and execution.

```text
+---------------------------------------------------------+
|                  User Application                       |
|            HIP-like API and kernel launches             |
|         hipMalloc, hipMemcpy, hipLaunchKernel           |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                    gcr_runtime                          |
|                       HIP Shim                          |
|          Translates HIP calls into GCR calls            |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                     gcr_device                          |
|                  Vulkan Compute Layer                  |
|       Pipelines, buffers, images, and dispatch          |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                      gcr_schd                           |
|                 Multi-GPU Scheduler                     |
|         Work distribution and load balancing            |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                       gcr_isa                           |
|                 gfx803 Shader Backend                   |
|             Generates SPIR-V for Vulkan                 |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                     gcr_loader                          |
|                  Vulkan Device Discovery                |
|          Enumerates GPUs and selects queues             |
+---------------------------------------------------------+
```

### What Each Component Does

**`gcr_runtime`**

This is the API layer that applications interact with. It provides HIP-like functions and translates application requests into operations that the underlying GCR components can handle.

**`gcr_device`**

This component manages Vulkan compute operations. It creates compute pipelines, allocates and manages buffers, records commands, and submits work to the GPU.

**`gcr_schd`**

This component distributes workloads across the available GPUs. It takes differences in PCIe bandwidth and other device capabilities into account when deciding where to execute work.

**`gcr_isa`**

This component handles shader generation for the gfx803 architecture. In the initial implementation, it generates SPIR-V rather than directly producing GCN machine code.

**`gcr_loader`**

This component discovers Vulkan devices, identifies suitable compute queues, and initializes the devices that the runtime will use.

## 4. Understanding the gfx803 Architecture

The RX 580 uses AMD's GCN 3rd-generation architecture, identified as gfx803.

The following table summarizes the relevant hardware characteristics.

| Property | Value |
|---|---|
| Architecture | GCN 3rd generation, gfx803 |
| Compute units | Up to 36 active CUs, depending on the specific GPU |
| Compute unit layout | 4 shader engines, 1 shader array per engine, 9 CUs per shader array |
| SIMDs per CU | 4 |
| Wavefront size | 64 threads |
| Maximum waves per SIMD | Up to 10 |
| Local data share (LDS) | Up to 64 KB per CU |
| VGPRs | 256 32-bit registers per SIMD, organized into register allocation units |
| SGPRs | Up to 102 32-bit scalar registers per wave, subject to hardware allocation rules |
| Workgroup shared memory | Uses LDS, subject to the hardware and Vulkan limits |

Actual resources available to a shader depend on the GPU model, driver, shader configuration, and resource allocation requirements.

### Mapping gfx803 to Vulkan Compute

Vulkan does not expose the GPU's GCN instruction set directly through its standard compute API.

Instead, GCR will generate SPIR-V, the intermediate representation used by Vulkan shaders. The RADV driver will then compile that SPIR-V into machine code suitable for the target GPU.

This means the initial implementation will follow three principles:

1. The `gcr_isa` component generates valid SPIR-V rather than raw GCN machine code.
2. RADV handles instruction selection, register allocation, instruction scheduling, and machine-code generation.
3. GCR relies on Mesa and RADV to produce executable code for the gfx803 GPUs.

This approach avoids the need to implement a complete GCN compiler in the first version of GCR.

The main trade-off is performance control. GCR will not have direct control over every generated instruction, and the quality of the resulting code will depend on RADV's compiler.

For the first release, the priority is to get correct compute execution working on all four GPUs. More advanced optimization can come later.

## 5. Multi-GPU Scheduling

The four GPUs have different PCIe connections. These differences affect how quickly data can move between the host system and each GPU.

| GPU | PCIe Slot | Connection | Stock KFD | Vulkan |
|---|---|---|---|---|
| GPU 0 | 01:00.0 | x8, directly connected to CPU | Yes | Yes |
| GPU 1 | 02:00.0 | x8, directly connected to CPU | Yes | Yes |
| GPU 2 | 03:00.0 | x4, connected through PCH | No, blocked by atomics requirement | Yes |
| GPU 3 | 05:00.0 | x1, connected through PCH riser | No, blocked by atomics requirement | Yes |

The scheduler should not assume that all four GPUs perform equally well.

GPU 0 and GPU 1 have wider PCIe connections and should generally be better suited for workloads that require frequent host-to-device transfers.

GPU 2 has a narrower x4 connection, while GPU 3 uses an x1 connection. These GPUs may still contribute useful compute capacity, particularly when a workload can remain in VRAM for long periods.

### Scheduling Strategy

The initial scheduler should use a simple, measurable approach.

1. Detect all available Vulkan compute devices.
2. Collect device properties and identify the PCIe topology where possible.
3. Run basic benchmarks to estimate each GPU's compute throughput and transfer bandwidth.
4. Assign work according to the measured performance of each GPU.
5. Keep intermediate data in GPU memory whenever possible to reduce PCIe traffic.
6. Synchronize results explicitly when work depends on output from another GPU.

The scheduler should use measured performance rather than relying only on PCIe link width. A narrower connection does not necessarily mean that a GPU will be slower for every workload.

For example, a workload that performs many calculations on data already stored in VRAM may benefit from GPU 3 despite its x1 connection.

By contrast, a workload that constantly transfers data between the CPU and GPU may perform poorly on that device.

## 6. Public API

GCR will initially expose a HIP-compatible API so that applications can use familiar memory management, device selection, and kernel-launch functions.

```c
hipError_t hipMalloc(void **ptr, size_t size);

hipError_t hipFree(void *ptr);

hipError_t hipMemcpy(
    void *dst,
    const void *src,
    size_t size,
    hipMemcpyKind kind
);

hipError_t hipSetDevice(int device);

hipError_t hipGetDeviceCount(int *count);

hipError_t hipLaunchKernel(
    const void *function,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    hipStream_t stream
);
```

These declarations describe the intended API surface. The implementation will need to define the exact types, error codes, and compatibility behavior.

### How the API Works

**Memory allocation**

`hipMalloc` allocates device memory through the Vulkan memory-management layer. The runtime must select a suitable Vulkan memory type and track the allocation.

`hipFree` releases the corresponding allocation and ensures that resources are not destroyed while commands are still using them.

**Memory transfers**

`hipMemcpy` transfers data between host and device memory or between other supported memory locations.

The implementation must respect the requested copy direction and handle synchronization where necessary.

**Device selection**

`hipSetDevice` selects the GPU that subsequent operations should use.

`hipGetDeviceCount` reports the number of GPUs successfully initialized and made available by GCR, rather than the number registered by KFD.

**Kernel execution**

`hipLaunchKernel` schedules a compute operation using the requested grid dimensions, workgroup dimensions, arguments, shared memory, and stream.

Because Vulkan does not execute arbitrary HIP kernel binaries directly, GCR will need a compatible shader representation or a translation and compilation process for supported kernels.

### Stream Handling

Each `hipStream_t` will represent an ordered sequence of compute operations associated with a specific device.

Internally, GCR can use Vulkan command buffers and queue submissions to implement this behavior. However, a stream should not be treated as a one-to-one replacement for a single command buffer.

Vulkan command buffers have their own recording and submission rules. GCR will need to manage command-buffer reuse, queue ordering, fences, semaphores, and resource lifetimes to provide predictable stream behavior.

## 7. Testing Plan

Testing should proceed in stages so that hardware discovery, memory management, shader execution, and multi-GPU scheduling can be debugged independently.

### Test 1: GPU Enumeration

Run the GCR diagnostic binary and verify that all four GPUs are detected through Vulkan.

Expected result: GCR reports four usable Vulkan devices, assuming the installed driver and system configuration support all four.

The test should also print each device's name, PCI identifier where available, queue capabilities, and relevant memory properties.

### Test 2: Buffer Allocation and Transfers

Test `hipMalloc`, `hipFree`, and `hipMemcpy` on each GPU individually.

The test should:

1. Allocate device memory.
2. Copy known data from host memory to the GPU.
3. Copy the data back to the host.
4. Compare the returned data against the original.
5. Release the allocation.

Expected result: Each GPU completes the round-trip transfer without data corruption or synchronization errors.

### Test 3: Compute Execution

Run a simple vector-addition kernel on each GPU.

The kernel should add two input arrays and write the results to an output array. The output should then be compared against a CPU reference implementation.

Expected result: Each GPU produces correct results through the Vulkan compute pipeline.

This test also verifies shader compilation, descriptor binding, workgroup configuration, dispatch, and completion synchronization.

### Test 4: Multi-GPU Execution

Divide a vector-addition workload across all four GPUs.

Each GPU should process its assigned portion of the input data. The runtime should collect the results and combine them into the final output.

Expected result: All four GPUs contribute to the computation, and the combined output matches the CPU reference.

The test should also verify that each device can operate independently and that resource management remains correct when multiple devices are active.

### Test 5: Sustained Compute on GPU 2 and GPU 3

Run repeated compute workloads on GPU 2 and GPU 3, which are not registered by the stock KFD driver.

Monitor the following:

- Shader compilation and pipeline creation.
- Command submission and completion.
- Memory allocation and release.
- Data transfer correctness.
- Synchronization behavior.
- Device loss or driver errors.
- Stability during extended execution.

Expected result: Both GPUs complete sustained workloads without unexpected failures.

This test is especially important because successful device enumeration alone does not prove that long-running compute workloads will work reliably.

## 8. Risk Assessment

| Risk | Mitigation |
|---|---|
| RADV generates code that is slower than expected for gfx803 | Prioritize correctness in the first version, then benchmark and optimize important workloads |
| SPIR-V cannot directly express every required low-level GCN operation | Use standard Vulkan compute features where possible and define a clear compatibility boundary |
| Multiple GPUs do not automatically share coherent memory | Maintain separate per-device allocations and implement explicit transfers between GPUs |
| GPU 3 is limited by its PCIe x1 connection | Keep data resident in VRAM and use benchmark-driven scheduling |
| HIP kernel binaries cannot be executed directly by Vulkan | Define a supported kernel input format and implement a translation or compilation pipeline |
| Vulkan device discovery succeeds but compute execution fails | Test pipeline creation, memory allocation, dispatch, and synchronization separately |
| Stream and event behavior differs from HIP | Implement and test explicit ordering and synchronization semantics |
| Driver or hardware limitations affect a particular GPU | Record device capabilities and disable unsupported features when necessary |

## 9. Future Work

Once the initial runtime can reliably enumerate all four GPUs, allocate memory, execute shaders, and distribute workloads, development can move toward performance and compatibility improvements.

### Shader Optimization

Improve generated SPIR-V patterns for gfx803 workloads. Benchmark register pressure, occupancy, memory access patterns, and workgroup configuration to identify performance bottlenecks.

### Graph Capture

Add support for capturing and replaying sequences of compute operations. This could reduce submission overhead for workloads that repeatedly execute the same sequence of GPU operations.

### PyTorch Integration

Build a custom PyTorch backend or integration layer that allows supported tensor operations to execute through GCR.

This will require more than a HIP-like memory API. The integration must also provide compatible kernel execution, tensor operations, synchronization, and memory handling.

### Direct GCN Code Generation

Investigate generating GCN machine code directly for selected performance-critical kernels.

This would offer more control over instruction selection and scheduling, but it would also require a substantially more complex compiler and validation process.

The direct code-generation path should be considered an optional optimization layer rather than a requirement for the initial Vulkan implementation.

## 10. Implementation Priorities

The recommended development order is:

1. Build `gcr_loader` and confirm that Vulkan exposes all four GPUs.
2. Implement per-device initialization and basic memory allocation in `gcr_device`.
3. Create a minimal SPIR-V compute pipeline and execute a simple kernel.
4. Implement the initial `gcr_runtime` API for memory management and dispatch.
5. Validate compute execution and memory transfers on every GPU.
6. Implement `gcr_schd` and distribute workloads across multiple GPUs.
7. Benchmark PCIe transfer performance and compute throughput to improve scheduling.
8. Add more advanced kernel translation, optimization, and framework integration.

This order keeps the initial implementation focused on a working compute path before adding more complex compatibility and optimization features.

## Conclusion

GCR aims to make all four RX 580 GPUs available for compute workloads by using Vulkan instead of relying on the stock KFD registration path.

The core idea is to use RADV for device access and shader compilation while GCR provides the application-facing API, memory management, kernel dispatch, and multi-GPU scheduling.

The most important first milestone is to confirm that all four GPUs can execute real Vulkan compute workloads reliably. Once that works, the runtime can be expanded with better scheduling, broader kernel support, and integrations with AI frameworks.

The Vulkan approach avoids the need to patch the Linux kernel as part of the initial design, but full HIP compatibility and high performance will require additional engineering beyond basic GPU enumeration and shader dispatch.