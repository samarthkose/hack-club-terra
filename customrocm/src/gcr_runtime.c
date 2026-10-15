/*
 * gcr — Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * HIP-like runtime implementation — maps to gcr_device.
 */

#include "gcr_runtime.h"
#include "gcr_device.h"
#include "gcr_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Global state ───────────────────────────────────────────────────────── */

#define GCR_MAX_DEVICES 8

typedef struct {
    gcr_device_t *devices[GCR_MAX_DEVICES];
    int device_count;
    int current_device;
    gcr_stream_t *default_stream;
    hipError_t last_error;
} gcr_runtime_state_t;

static gcr_runtime_state_t g_state = {
    .device_count = 0,
    .current_device = 0,
    .default_stream = NULL,
    .last_error = hipSuccess,
};

/* ─── Initialization (lazy) ──────────────────────────────────────────────── */

static hipError_t runtime_init(void) {
    if (g_state.device_count > 0) return hipSuccess;

    gcr_loader_state_t loader;
    int rc = gcr_loader_init(&loader);
    if (rc != 0) {
        g_state.last_error = hipErrorNoDevice;
        return hipErrorNoDevice;
    }

    g_state.device_count = (int)loader.device_count;
    for (int i = 0; i < g_state.device_count; i++) {
        g_state.devices[i] = gcr_device_create(i);
        if (!g_state.devices[i]) {
            g_state.last_error = hipErrorInvalidDevice;
            return hipErrorInvalidDevice;
        }
    }

    /* Create default stream on device 0 */
    g_state.default_stream = gcr_device_create_stream(g_state.devices[0]);

    gcr_loader_cleanup(&loader);
    fprintf(stderr, "gcr_runtime: initialized with %d device(s)\n", g_state.device_count);
    return hipSuccess;
}

/* ─── Device management ──────────────────────────────────────────────────── */

hipError_t hipGetDeviceCount(int *count) {
    if (!count) return hipErrorInvalidValue;
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;
    *count = g_state.device_count;
    return hipSuccess;
}

hipError_t hipSetDevice(int device) {
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;
    if (device < 0 || device >= g_state.device_count) {
        g_state.last_error = hipErrorInvalidDevice;
        return hipErrorInvalidDevice;
    }
    g_state.current_device = device;
    return hipSuccess;
}

hipError_t hipGetDevice(int *device) {
    if (!device) return hipErrorInvalidValue;
    *device = g_state.current_device;
    return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device) {
    if (!prop) return hipErrorInvalidValue;
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;
    if (device < 0 || device >= g_state.device_count) {
        g_state.last_error = hipErrorInvalidDevice;
        return hipErrorInvalidDevice;
    }

    const gcr_device_info_t *info = gcr_device_get_info(g_state.devices[device]);
    if (!info) return hipErrorInvalidDevice;

    memset(prop, 0, sizeof(*prop));
    strncpy(prop->name, info->device_name, sizeof(prop->name) - 1);
    prop->totalGlobalMem = (size_t)info->heap_size;
    prop->multiProcessorCount = 36;  /* gfx803: 36 CUs */
    prop->clockRate = 1366000;       /* RX 580: ~1366 MHz */
    prop->major = 8;
    prop->minor = 3;
    prop->sharedMemPerBlock = 64 * 1024;
    prop->regsPerBlock = 65536;
    prop->warpSize = 64;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 1024;
    prop->maxGridSize[0] = 65535;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;

    return hipSuccess;
}

/* ─── Memory management ──────────────────────────────────────────────────── */

/*
 * Internal buffer wrapper — we need to track which device owns it.
 */
typedef struct {
    gcr_device_t *dev;
    gcr_buffer_t *buf;
    void *host_ptr;     /* mapped host pointer if host-visible */
    size_t size;
    uint32_t magic;
} gcr_alloc_t;

#define GCR_ALLOC_MAGIC 0x67637241  /* "gcrA" */

hipError_t hipMalloc(void **ptr, size_t size) {
    if (!ptr || size == 0) return hipErrorInvalidValue;
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    gcr_device_t *dev = g_state.devices[g_state.current_device];
    if (!dev) return hipErrorInvalidDevice;

    gcr_alloc_t *alloc = calloc(1, sizeof(gcr_alloc_t));
    if (!alloc) return hipErrorOutOfMemory;

    /* Allocate host-visible buffer for HIP compatibility (simpler mapping) */
    alloc->buf = gcr_device_alloc_buffer(dev, size, true);
    alloc->dev = dev;
    alloc->size = size;
    alloc->magic = GCR_ALLOC_MAGIC;

    if (!alloc->buf) {
        free(alloc);
        g_state.last_error = hipErrorOutOfMemory;
        return hipErrorOutOfMemory;
    }

    *ptr = (void *)alloc;
    return hipSuccess;
}

hipError_t hipFree(void *ptr) {
    if (!ptr) return hipSuccess;
    gcr_alloc_t *alloc = (gcr_alloc_t *)ptr;
    if (alloc->magic != GCR_ALLOC_MAGIC) {
        g_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    gcr_device_free_buffer(alloc->dev, alloc->buf);
    alloc->magic = 0;
    free(alloc);
    return hipSuccess;
}

/*
 * Helper: get gcr_alloc_t from a user pointer.
 */
static gcr_alloc_t *get_alloc(void *ptr) {
    if (!ptr) return NULL;
    gcr_alloc_t *alloc = (gcr_alloc_t *)ptr;
    if (alloc->magic != GCR_ALLOC_MAGIC) return NULL;
    return alloc;
}

hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) {
    if (!dst || !src || size == 0) return hipErrorInvalidValue;
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    switch (kind) {
        case hipMemcpyHostToDevice: {
            gcr_alloc_t *alloc = get_alloc(dst);
            if (!alloc) return hipErrorInvalidValue;
            return gcr_device_memcpy_h2d(alloc->dev, alloc->buf, src, 0, size) == 0
                ? hipSuccess : hipErrorInvalidValue;
        }
        case hipMemcpyDeviceToHost: {
            gcr_alloc_t *alloc = get_alloc((void *)src);
            if (!alloc) return hipErrorInvalidValue;
            return gcr_device_memcpy_d2h(alloc->dev, alloc->buf, dst, 0, size) == 0
                ? hipSuccess : hipErrorInvalidValue;
        }
        case hipMemcpyDeviceToDevice: {
            gcr_alloc_t *src_alloc = get_alloc((void *)src);
            gcr_alloc_t *dst_alloc = get_alloc(dst);
            if (!src_alloc || !dst_alloc) return hipErrorInvalidValue;
            return gcr_device_memcpy_d2d(dst_alloc->dev, dst_alloc->buf, 0,
                                         src_alloc->buf, 0, size) == 0
                ? hipSuccess : hipErrorInvalidValue;
        }
        case hipMemcpyHostToHost:
            memcpy(dst, src, size);
            return hipSuccess;
        default:
            return hipErrorInvalidValue;
    }
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream) {
    /* For now, just do synchronous copy — stream is ignored */
    (void)stream;
    return hipMemcpy(dst, src, size, kind);
}

hipError_t hipMemset(void *dst, int value, size_t count) {
    if (!dst) return hipErrorInvalidValue;
    gcr_alloc_t *alloc = get_alloc(dst);
    if (!alloc) return hipErrorInvalidValue;

    /* Allocate temp buffer, memset, copy */
    void *tmp = malloc(count);
    if (!tmp) return hipErrorOutOfMemory;
    memset(tmp, value, count);

    hipError_t err = gcr_device_memcpy_h2d(alloc->dev, alloc->buf, tmp, 0, count) == 0
        ? hipSuccess : hipErrorInvalidValue;
    free(tmp);
    return err;
}

hipError_t hipMemsetAsync(void *dst, int value, size_t count, hipStream_t stream) {
    (void)stream;
    return hipMemset(dst, value, count);
}

/* ─── Stream management ──────────────────────────────────────────────────── */

hipError_t hipStreamCreate(hipStream_t *stream) {
    if (!stream) return hipErrorInvalidValue;
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    gcr_device_t *dev = g_state.devices[g_state.current_device];
    gcr_stream_t *s = gcr_device_create_stream(dev);
    if (!s) return hipErrorOutOfMemory;

    *stream = (hipStream_t)s;
    return hipSuccess;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    if (!stream) return hipSuccess;
    gcr_stream_t *s = (gcr_stream_t *)stream;
    /* TODO: need device handle — store it or track differently */
    free(s);
    return hipSuccess;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    if (!stream) {
        /* Sync default stream */
        if (g_state.default_stream)
            return gcr_stream_wait(g_state.default_stream) == 0 ? hipSuccess : hipErrorUnknown;
        return hipSuccess;
    }
    gcr_stream_t *s = (gcr_stream_t *)stream;
    return gcr_stream_wait(s) == 0 ? hipSuccess : hipErrorUnknown;
}

hipError_t hipStreamQuery(hipStream_t stream, int *status) {
    if (!status) return hipErrorInvalidValue;
    /* TODO: non-blocking query */
    *status = 1; /* assume completed */
    return hipSuccess;
}

/* ─── Kernel launch ──────────────────────────────────────────────────────── */

/*
 * For now, kernel launch is a stub — the actual dispatch goes through
 * gcr_stream_dispatch once we have a pipeline. The `function` parameter
 * should be a pointer to a gcr_pipeline_t (or SPIR-V module).
 *
 * TODO: wire up gcr_isa to generate SPIR-V, create pipeline, dispatch.
 */
hipError_t hipLaunchKernel(const void *function,
                           dim3 gridDim,
                           dim3 blockDim,
                           void **args,
                           size_t sharedMemBytes,
                           hipStream_t stream) {
    (void)function;
    (void)gridDim;
    (void)blockDim;
    (void)args;
    (void)sharedMemBytes;
    (void)stream;

    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    fprintf(stderr, "gcr_runtime: hipLaunchKernel called (stub — not yet wired to gcr_device)\n");
    g_state.last_error = hipErrorNotReady;
    return hipErrorNotReady;
}

/* ─── Synchronization ────────────────────────────────────────────────────── */

hipError_t hipDeviceSynchronize(void) {
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    if (g_state.default_stream)
        return gcr_stream_wait(g_state.default_stream) == 0 ? hipSuccess : hipErrorUnknown;
    return hipSuccess;
}

hipError_t hipDeviceSynchronizeAll(void) {
    hipError_t err = runtime_init();
    if (err != hipSuccess) return err;

    for (int i = 0; i < g_state.device_count; i++) {
        /* TODO: sync all streams on each device */
        (void)i;
    }
    return hipSuccess;
}

/* ─── Error handling ─────────────────────────────────────────────────────── */

const char *hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess:              return "hipSuccess";
        case hipErrorInvalidValue:    return "hipErrorInvalidValue";
        case hipErrorOutOfMemory:     return "hipErrorOutOfMemory";
        case hipErrorNotInitialized:  return "hipErrorNotInitialized";
        case hipErrorDeinitialized:   return "hipErrorDeinitialized";
        case hipErrorNoDevice:        return "hipErrorNoDevice";
        case hipErrorInvalidDevice:   return "hipErrorInvalidDevice";
        case hipErrorInvalidImage:    return "hipErrorInvalidImage";
        case hipErrorInvalidArgument: return "hipErrorInvalidArgument";
        case hipErrorInvalidHandle:   return "hipErrorInvalidHandle";
        case hipErrorNotReady:        return "hipErrorNotReady";
        case hipErrorLaunchTimeout:   return "hipErrorLaunchTimeout";
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
        case hipErrorLaunchFailure:   return "hipErrorLaunchFailure";
        default:                      return "hipErrorUnknown";
    }
}

hipError_t hipGetLastError(void) {
    hipError_t err = g_state.last_error;
    g_state.last_error = hipSuccess;
    return err;
}

hipError_t hipPeekAtLastError(void) {
    return g_state.last_error;
}