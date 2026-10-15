/*
 * gcr — Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * Loader implementation: Vulkan instance/device discovery.
 */

#include "gcr_loader.h"

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* AMD vendor ID */
#define GCR_VENDOR_AMD 0x1002
/* RX 580 / Ellesmere PCI device ID */
#define GCR_DEVICE_ELLESMERE 0x67df

static VkInstance g_instance = VK_NULL_HANDLE;
static uint32_t g_physical_device_count = 0;
static VkPhysicalDevice g_physical_devices[GCR_MAX_DEVICES];

int gcr_loader_init(gcr_loader_state_t *state) {
    if (!state) return -1;
    memset(state, 0, sizeof(*state));

    /* Create Vulkan instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "gcr",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "gcr",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };

    VkResult res = vkCreateInstance(&create_info, NULL, &g_instance);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gcr_loader: vkCreateInstance failed: %d\n", res);
        return -2;
    }

    /* Enumerate physical devices */
    res = vkEnumeratePhysicalDevices(g_instance, &g_physical_device_count, NULL);
    if (res != VK_SUCCESS || g_physical_device_count == 0) {
        fprintf(stderr, "gcr_loader: no Vulkan devices found\n");
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
        return -3;
    }

    if (g_physical_device_count > GCR_MAX_DEVICES) {
        g_physical_device_count = GCR_MAX_DEVICES;
    }

    res = vkEnumeratePhysicalDevices(g_instance, &g_physical_device_count,
                                     g_physical_devices);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gcr_loader: vkEnumeratePhysicalDevices failed: %d\n", res);
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
        return -4;
    }

    /* Filter and populate device info */
    state->device_count = 0;
    for (uint32_t i = 0; i < g_physical_device_count && state->device_count < GCR_MAX_DEVICES; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_physical_devices[i], &props);

        /* Only include AMD discrete GPUs (gfx803 target) */
        if (props.vendorID != GCR_VENDOR_AMD) continue;
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;

        gcr_device_info_t *info = &state->devices[state->device_count];
        info->api_version = props.apiVersion;
        info->driver_version = props.driverVersion;
        info->vendor_id = props.vendorID;
        info->device_id = props.deviceID;
        info->is_amdgpu = true;
        strncpy(info->device_name, props.deviceName, sizeof(info->device_name) - 1);
        info->device_name[sizeof(info->device_name) - 1] = '\0';

        /* Extract UUID first 4 bytes */
        memcpy(&info->device_uuid, props.pipelineCacheUUID, sizeof(uint32_t));

        /* Query memory heaps for VRAM */
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(g_physical_devices[i], &mem_props);
        info->heap_size = 0;
        for (uint32_t h = 0; h < mem_props.memoryHeapCount; h++) {
            if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                if (mem_props.memoryHeaps[h].size > info->heap_size) {
                    info->heap_size = mem_props.memoryHeaps[h].size;
                }
            }
        }

        /* Find compute queue family */
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g_physical_devices[i],
                                                  &queue_family_count, NULL);
        VkQueueFamilyProperties queue_families[64];
        if (queue_family_count > 64) queue_family_count = 64;
        vkGetPhysicalDeviceQueueFamilyProperties(g_physical_devices[i],
                                                  &queue_family_count,
                                                  queue_families);

        info->compute_queue_family = UINT32_MAX;
        for (uint32_t q = 0; q < queue_family_count; q++) {
            if (queue_families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                info->compute_queue_family = q;
                break;
            }
        }

        if (info->compute_queue_family == UINT32_MAX) {
            fprintf(stderr, "gcr_loader: device %u has no compute queue\n", i);
            continue; /* skip devices without compute */
        }

        state->device_count++;
    }

    fprintf(stderr, "gcr_loader: found %u AMD discrete GPU(s)\n", state->device_count);
    for (uint32_t i = 0; i < state->device_count; i++) {
        fprintf(stderr, "  [%u] %s (0x%04x:0x%04x) VRAM=%lu MB\n",
                i, state->devices[i].device_name,
                state->devices[i].vendor_id, state->devices[i].device_id,
                state->devices[i].heap_size / (1024 * 1024));
    }

    return 0;
}

int gcr_loader_get_device(const gcr_loader_state_t *state,
                          uint32_t index,
                          gcr_device_info_t *info) {
    if (!state || !info || index >= state->device_count) return -1;
    *info = state->devices[index];
    return 0;
}

void gcr_loader_cleanup(gcr_loader_state_t *state) {
    if (g_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
    }
    if (state) {
        memset(state, 0, sizeof(*state));
    }
    g_physical_device_count = 0;
}