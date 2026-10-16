/*
 * gcr - Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * Main entry point: enumerate devices, print status.
 */

#include "gcr_loader.h"
#include <stdio.h>

int main(void) {
    gcr_loader_state_t loader;
    int rc = gcr_loader_init(&loader);
    if (rc != 0) {
        fprintf(stderr, "gcr_loader_init failed: %d\n", rc);
        return 1;
    }

    printf("gcr loader initialized: %u device(s)\n", loader.device_count);
    for (uint32_t i = 0; i < loader.device_count; i++) {
        printf("  [%u] %s\n", i, loader.devices[i].device_name);
        printf("      PCI: %04x:%04x  VRAM: %lu MB  compute_queue: %u\n",
        loader.devices[i].vendor_id,
        loader.devices[i].device_id,
        loader.devices[i].heap_size / (1024 * 1024),
        loader.devices[i].compute_queue_family);
    }
    gcr_loader_cleanup(&loader);
    return 0;
}