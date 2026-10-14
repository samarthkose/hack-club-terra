/*
 * gcr — Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * HIP-like runtime API — maps to gcr_device calls.
 */

#ifndef GCR_RUNTIME_H
#define GCR_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ─── HIP-compatible types ───────────────────────────────────────────────── */

typedef enum {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorDeinitialized = 4,
    hipErrorNoDevice = 5,
    hipErrorInvalidDevice = 6,
    hipErrorInvalidImage = 7,
    hipErrorInvalidArgument = 8,
    hipErrorInvalidHandle = 9,
    hipErrorNotReady = 10,
    hipErrorLaunchTimeout = 11,
    hipErrorLaunchOutOfResources = 12,
    hipErrorLaunchFailure = 13,
    hipErrorUnknown = 999,
} hipError_t;

typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
} hipMemcpyKind;

typedef struct dim3_st { uint32_t x, y, z; } dim3;

/* Opaque handles */
typedef struct gcr_device_impl *hipDevice_t;
typedef struct gcr_buffer_impl *hipDeviceptr_t;
typedef struct gcr_stream_impl *hipStream_t;

/* ─── Device management ──────────────────────────────────────────────────── */

/*
 * Get the number of available GPUs.
 * Returns hipSuccess on success.
 */
hipError_t hipGetDeviceCount(int *count);

/*
 * Set the current device for subsequent operations.
 */
hipError_t hipSetDevice(int device);

/*
 * Get the current device.
 */
hipError_t hipGetDevice(int *device);

/*
 * Get device properties (name, totalGlobalMem, etc.).
 */
typedef struct {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int clockRate;           /* kHz */
    int major, minor;        /* compute capability */
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
} hipDeviceProp_t;

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device);

/* ─── Memory management ──────────────────────────────────────────────────── */

/*
 * Allocate memory on the current device.
 */
hipError_t hipMalloc(void **ptr, size_t size);

/*
 * Free memory.
 */
hipError_t hipFree(void *ptr);

/*
 * Copy memory between host and device (or device to device).
 */
hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind);

/*
 * Copy memory asynchronously.
 */
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream);

/*
 * Set memory to a value.
 */
hipError_t hipMemset(void *dst, int value, size_t count);

/*
 * Set memory to a value asynchronously.
 */
hipError_t hipMemsetAsync(void *dst, int value, size_t count, hipStream_t stream);

/* ─── Stream management ──────────────────────────────────────────────────── */

/*
 * Create a stream.
 */
hipError_t hipStreamCreate(hipStream_t *stream);

/*
 * Destroy a stream.
 */
hipError_t hipStreamDestroy(hipStream_t stream);

/*
 * Wait for all operations in a stream to complete.
 */
hipError_t hipStreamSynchronize(hipStream_t stream);

/*
 * Query if a stream has completed.
 */
hipError_t hipStreamQuery(hipStream_t stream, int *status);

/* ─── Kernel launch ──────────────────────────────────────────────────────── */

/*
 * Launch a compute kernel.
 *
 * @param function: pointer to the kernel function (SPIR-V module or encoded shader)
 * @param gridDim: grid dimensions (number of workgroups)
 * @param blockDim: block dimensions (workgroup size)
 * @param args: array of kernel arguments (each is a pointer to the argument data)
 * @param sharedMemBytes: dynamic shared memory size (not used in Vulkan path)
 * @param stream: stream to launch on (NULL for default stream)
 */
hipError_t hipLaunchKernel(const void *function,
                           dim3 gridDim,
                           dim3 blockDim,
                           void **args,
                           size_t sharedMemBytes,
                           hipStream_t stream);

/* ─── Synchronization ────────────────────────────────────────────────────── */

/*
 * Wait for all work on the current device to complete.
 */
hipError_t hipDeviceSynchronize(void);

/*
 * Wait for all work on all devices to complete.
 */
hipError_t hipDeviceSynchronizeAll(void);

/* ─── Error handling ─────────────────────────────────────────────────────── */

/*
 * Get a string description of an error code.
 */
const char *hipGetErrorString(hipError_t error);

/*
 * Get the last error.
 */
hipError_t hipGetLastError(void);

/*
 * Clear the last error.
 */
hipError_t hipPeekAtLastError(void);

#endif /* GCR_RUNTIME_H */