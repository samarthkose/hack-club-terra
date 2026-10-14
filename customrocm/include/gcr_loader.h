/*
 * gcr — Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * Loader: Vulkan instance/device discovery, enumerate all four GPUs.
 */

#ifndef GCR_LOADER_H
#define GCR_LOADER_H

#include <stdint.h>
#include <stdbool.h>

#define GCR_MAX_DEVICES 8

typedef struct {
    uint32_t api_version;       /* Vulkan API version */
    uint32_t driver_version;    /* driver version */
    uint32_t vendor_id;         /* 0x1002 = AMD */
    uint32_t device_id;         /* PCI device ID (0x67df for RX 580) */
    uint32_t device_uuid;       /* first 4 bytes of UUID */
    char     device_name[256];  /* human-readable name */
    uint64_t heap_size;         /* dedicated VRAM size (bytes) */
    uint32_t compute_queue_family; /* queue family index for compute */
    bool     is_amdgpu;         /* true if RADV/amdgpu driver */
} gcr_device_info_t;

typedef struct {
    uint32_t device_count;
    gcr_device_info_t devices[GCR_MAX_DEVICES];
} gcr_loader_state_t;

/*
 * Initialize Vulkan and enumerate all available devices.
 * Returns 0 on success, negative on error.
 */
int gcr_loader_init(gcr_loader_state_t *state);

/*
 * Get device info for a specific index.
 * Returns 0 on success, negative on error.
 */
int gcr_loader_get_device(const gcr_loader_state_t *state,
                          uint32_t index,
                          gcr_device_info_t *info);

/*
 * Cleanup Vulkan loader.
 */
void gcr_loader_cleanup(gcr_loader_state_t *state);

#endif /* GCR_LOADER_H */