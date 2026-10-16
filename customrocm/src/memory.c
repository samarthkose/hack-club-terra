#include "croc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t find_memory_type(croc_device* device, uint32_t type_bits, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < device->memory_properties.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) && (device->memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return UINT32_MAX;
}

/*
* Host-visible placement policy (2026-10-13): RADV on this host exposes
*   type 2: HOST_VISIBLE|COHERENT on the GTT (system RAM) heap
*   type 3/4: HOST_VISIBLE|COHERENT|DEVICE_LOCAL on the VRAM heap (Above-4G BAR)
* The first-match scan above picked type 2, so "VRAM" buffers silently lived
* in system RAM (memtype_probe: vram_used delta = 0 during a 1 GiB alloc;
* bw_test: identical ~8.9 GB/s "H2D" on every link including x1). Prefer the
* device-local host-visible type so host-visible == real VRAM via the BAR. */

static uint32_t find_host_visible_type(croc_device* device, uint32_t type_bits) {
    uint32_t t = find_memory_type(device, type_bits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
if (t != UINT32_MAX) return t;
return find_memory_type(device, type_bits,
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

croc_result croc_buffer_allocate(croc_device* device, size_t size, croc_buffer** buffer) {
    if (!device || size == 0 || !buffer) return CROC_ERR_INVALID_ARG;

croc_buffer* buf = calloc(1, sizeof(croc_buffer));
if (!buf) return CROC_ERR_OUT_OF_MEMORY;

buf->device = device;
buf->size = size;

VkBufferCreateInfo buffer_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = size,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
};
if (vkCreateBuffer(device->device, &buffer_info, NULL, &buf->buffer) != VK_SUCCESS) {
    free(buf);
    return CROC_ERR_OUT_OF_RESOURCES;
}

VkMemoryRequirements mem_reqs;
vkGetBufferMemoryRequirements(device->device, buf->buffer, &mem_reqs);

uint32_t memory_type = find_host_visible_type(device, mem_reqs.memoryTypeBits);
if (memory_type == UINT32_MAX) memory_type = find_memory_type(device, mem_reqs.memoryTypeBits, 0);
if (memory_type == UINT32_MAX) {
    vkDestroyBuffer(device->device, buf->buffer, NULL);
    free(buf);
    return CROC_ERR_OUT_OF_RESOURCES;
}

VkMemoryAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = mem_reqs.size,
    .memoryTypeIndex = memory_type,
};
if (vkAllocateMemory(device->device, &alloc_info, NULL, &buf->memory) != VK_SUCCESS) {
    vkDestroyBuffer(device->device, buf->buffer, NULL);
    free(buf);
    return CROC_ERR_OUT_OF_RESOURCES;
}

if (vkBindBufferMemory(device->device, buf->buffer, buf->memory, 0) != VK_SUCCESS) {
    vkFreeMemory(device->device, buf->memory, NULL);
    vkDestroyBuffer(device->device, buf->buffer, NULL);
    free(buf);
    return CROC_ERR_OUT_OF_RESOURCES;
}

*buffer = buf;
return CROC_OK;
}

void croc_buffer_release(croc_buffer* buffer) {
    if (!buffer) return;
    VkDevice dev = buffer->device->device;
    if (buffer->mapped) vkUnmapMemory(dev, buffer->memory);
    if (buffer->buffer) vkDestroyBuffer(dev, buffer->buffer, NULL);
    if (buffer->memory) vkFreeMemory(dev, buffer->memory, NULL);
    free(buffer);
}

croc_result croc_buffer_map(croc_buffer* buffer, void** data) {
    if (!buffer || !data) return CROC_ERR_INVALID_ARG;
    if (buffer->mapped) {
        *data = buffer->mapped;
        return CROC_OK;
    }
    if (vkMapMemory(buffer->device->device, buffer->memory, 0, buffer->size, 0, &buffer->mapped) != VK_SUCCESS)
    return CROC_ERR_MAP_FAILED;
    *data = buffer->mapped;
    return CROC_OK;
}

croc_result croc_buffer_unmap(croc_buffer* buffer) {
    if (!buffer) return CROC_ERR_INVALID_ARG;
    if (buffer->mapped) {
        vkUnmapMemory(buffer->device->device, buffer->memory);
        buffer->mapped = NULL;
    }
    return CROC_OK;
}
