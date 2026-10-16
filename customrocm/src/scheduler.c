#include "croc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t device_index;
    uint32_t offset;
    uint32_t length;
} layer_shard;

croc_result croc_layer_plan(croc_device* const* devices, uint32_t device_count,
size_t total_size, uint32_t* shard_sizes) {
    if (!devices || device_count == 0 || total_size == 0 || !shard_sizes)
return CROC_ERR_INVALID_ARG;

uint32_t base = (uint32_t)(total_size / device_count);
uint32_t remainder = (uint32_t)(total_size % device_count);

for (uint32_t i = 0; i < device_count; i++) {
    shard_sizes[i] = base;
    if (i < remainder) shard_sizes[i]++;
}

return CROC_OK;
}

croc_result croc_layer_dispatch(croc_kernel* const* kernels, uint32_t device_count,
    const uint32_t* shard_sizes, croc_stream* const* streams) {
        if (!kernels || device_count == 0 || !shard_sizes || !streams)
        return CROC_ERR_INVALID_ARG;

        for (uint32_t i = 0; i < device_count; i++) {
            uint32_t workgroup_count[] = { (shard_sizes[i] + 255) / 256, 1, 1 };
            croc_result r = croc_kernel_dispatch(kernels[i], workgroup_count, streams[i]);
            if (r != CROC_OK) return r;
        }

        return CROC_OK;
    }

the window is focused. only on vscode so you prob wont see anything, Im just pushing and saving to github rn