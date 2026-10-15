/*
 * CROC — loader.c
 *
 * Vulkan instance + physical device enumeration, logical device creation,
 * device property population (incl. PCIe topology via VK_EXT_pci_bus_info
 * + sysfs) and result-code strings.
 *
 * Filters to AMD discrete GPUs only (RADV), so lavapipe/llvmpipe CPU
 * devices never appear as CROC devices.
 */

#include "croc_internal.h"

#include <dirent.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CROC_VENDOR_AMD 0x1002
#define CROC_DEVICE_ELLESMERE 0x67df   /* RX 470/480/570/580 */

/* ─── result strings ─────────────────────────────────────────────────────── */

const char* croc_result_string(croc_result result) {
    switch (result) {
        case CROC_OK:                       return "CROC_OK";
        case CROC_ERR_UNKNOWN:              return "CROC_ERR_UNKNOWN";
        case CROC_ERR_INVALID_ARG:          return "CROC_ERR_INVALID_ARG";
        case CROC_ERR_OUT_OF_MEMORY:        return "CROC_ERR_OUT_OF_MEMORY";
        case CROC_ERR_INITIALIZATION_FAILED:return "CROC_ERR_INITIALIZATION_FAILED";
        case CROC_ERR_NO_DEVICE:            return "CROC_ERR_NO_DEVICE";
        case CROC_ERR_NOT_SUPPORTED:        return "CROC_ERR_NOT_SUPPORTED";
        case CROC_ERR_OUT_OF_RESOURCES:     return "CROC_ERR_OUT_OF_RESOURCES";
        case CROC_ERR_LAUNCH_FAILED:        return "CROC_ERR_LAUNCH_FAILED";
        case CROC_ERR_SYNC_FAILED:          return "CROC_ERR_SYNC_FAILED";
        case CROC_ERR_MAP_FAILED:           return "CROC_ERR_MAP_FAILED";
        case CROC_ERR_UNMAP_FAILED:         return "CROC_ERR_UNMAP_FAILED";
        case CROC_ERR_COPY_FAILED:          return "CROC_ERR_COPY_FAILED";
        case CROC_ERR_CREATE_KERNEL_FAILED: return "CROC_ERR_CREATE_KERNEL_FAILED";
        case CROC_ERR_SET_ARG_FAILED:       return "CROC_ERR_SET_ARG_FAILED";
        case CROC_ERR_DISPATCH_FAILED:      return "CROC_ERR_DISPATCH_FAILED";
        default:                            return "CROC_ERR_?";
    }
}

/* ─── sysfs helpers ──────────────────────────────────────────────────────── */

static int read_sysfs_int(const char* path, int* out) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v, n = fscanf(f, "%d", &v);
    fclose(f);
    if (n != 1) return -1;
    *out = v;
    return 0;
}

/* "8 GT/s" -> PCIe generation (2.5/5/8/16/32 GT/s -> gen 1..5) */
static uint32_t sysfs_link_gen(const char* devdir) {
    char path[PATH_MAX + 32], buf[64];
    snprintf(path, sizeof(path), "%s/current_link_speed", devdir);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    double gt = 0;
    if (sscanf(buf, "%lf", &gt) != 1 || gt <= 0) return 0;
    if (gt <= 2.5)  return 1;
    if (gt <= 5.0)  return 2;
    if (gt <= 8.0)  return 3;
    if (gt <= 16.0) return 4;
    return 5;
}

/* pp_dpm_sclk lines like "0: 300Mhz" / "1: 1366Mhz" -> max MHz */
static uint32_t sysfs_max_sclk(const char* devdir) {
    char path[PATH_MAX + 32];
    snprintf(path, sizeof(path), "%s/pp_dpm_sclk", devdir);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    char line[128];
    uint32_t max_mhz = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned mhz = 0;
        char* p = strchr(line, ':');
        if (p && sscanf(p + 1, " %u", &mhz) == 1 && mhz > max_mhz) max_mhz = mhz;
    }
    fclose(f);
    return max_mhz;
}

/*
 * Tier classification: walk sysfs ancestry of the PCI device and find the
 * nearest 0000:00:xx.y ancestor (a CPU root port or PCH root port).
 *   00:01.0 / 00:01.1  -> CPU-attached (fast)
 *   any other 00:xx.y  -> behind PCH (slow lane for H2D/D2H)
 */
static void detect_topology(croc_device* dev) {
    dev->properties.tier = CROC_TIER_UNKNOWN;
    dev->properties.link_width = 0;
    dev->properties.link_gen = 0;
    dev->properties.max_clock_mhz = 0;

    if (!dev->has_pci_bus_info) return;

    char devdir[PATH_MAX];
    snprintf(devdir, sizeof(devdir), "/sys/bus/pci/devices/0000:%02x:%02x.%x",
             dev->pci_bus, dev->pci_device, dev->pci_function);

    int w = 0;
    char link_w[PATH_MAX + 32];
    snprintf(link_w, sizeof(link_w), "%s/current_link_width", devdir);
    if (read_sysfs_int(link_w, &w) == 0 && w > 0) dev->properties.link_width = (uint32_t)w;
    dev->properties.link_gen = sysfs_link_gen(devdir);
    dev->properties.max_clock_mhz = sysfs_max_sclk(devdir);

    char real[PATH_MAX];
    if (!realpath(devdir, real)) return;

    /* tokenize path, find nearest ancestor whose BDF is on bus 00 */
    char work[PATH_MAX];
    snprintf(work, sizeof(work), "%s", real);
    for (char* slash = strrchr(work, '/'); slash; ) {
        *slash = '\0';
        unsigned b = 0, d = 0, f = 0;
        char* base = strrchr(work, '/');
        base = base ? base + 1 : work;
        if (sscanf(base, "0000:%x:%x.%x", &b, &d, &f) == 3 && b == 0) {
            if (d == 0x01 && f <= 0x01)
                dev->properties.tier = CROC_TIER_CPU_ATTACHED;
            else
                dev->properties.tier = CROC_TIER_PCH;
            return;
        }
        slash = strrchr(work, '/');
    }
}

/* System-wide count of GPU nodes exposed by KFD (0 = KFD unavailable). */
static uint32_t kfd_gpu_node_count(void) {
    DIR* d = opendir("/sys/class/kfd/kfd/topology/nodes");
    if (!d) return 0;
    uint32_t n = 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL)
        if (e->d_name[0] != '.') n++;
    closedir(d);
    return n > 0 ? n - 1 : 0;   /* node 0 is the CPU node */
}

/* ─── device property population ─────────────────────────────────────────── */

static void populate_properties(croc_device* dev) {
    const VkPhysicalDeviceProperties* p = &dev->vk_props;
    croc_device_properties* pr = &dev->properties;

    pr->vendor_id = p->vendorID;
    pr->device_id = p->deviceID;
    pr->subsystem_id = 0;
    pr->subsystem_vendor_id = 0;

    if (p->deviceID == CROC_DEVICE_ELLESMERE) {
        pr->gfx_target = 0x803;        /* gfx803 */
        pr->cu_count = 36;             /* RX 580: 36 CUs */
    } else {
        pr->gfx_target = 0;
        pr->cu_count = 0;
    }

    const VkPhysicalDeviceLimits* L = &p->limits;
    pr->vram_size = 0;
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(dev->physical_device, &mp);
        dev->memory_properties = mp;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if ((mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                mp.memoryHeaps[h].size > pr->vram_size)
                pr->vram_size = mp.memoryHeaps[h].size;
    }

    pr->max_compute_units = pr->cu_count;
    pr->max_workgroup_size = L->maxComputeWorkGroupInvocations;
    pr->max_workgroup_count = L->maxComputeWorkGroupCount[0];
    pr->max_mem_alloc_size = 0;   /* bytes; not exposed as a single limit */
    pr->max_constant_buffer_size = (uint32_t)L->maxUniformBufferRange;
    pr->max_constant_args = L->maxBoundDescriptorSets;
    pr->max_push_constants_size = L->maxPushConstantsSize;
    pr->max_sampler_count = L->maxSamplerAllocationCount;
    pr->max_storage_buffers = L->maxPerStageDescriptorStorageBuffers;
    pr->max_storage_images = L->maxPerStageDescriptorStorageImages;
    pr->max_uniform_buffers = L->maxPerStageDescriptorUniformBuffers;
    pr->max_color_attachments = L->maxColorAttachments;
    pr->max_depth_stencil_attachments = 1;
    pr->max_viewports = L->maxViewports;
    pr->max_scissor_rects = L->maxViewports;   /* no dedicated scissor limit in Vulkan */
    pr->max_draw_indirect_count = L->maxDrawIndirectCount;
    pr->max_fragment_atomic_counters = 0;
    pr->max_compute_atomic_counters = 0;
    pr->max_compute_shared_memory_size = L->maxComputeSharedMemorySize;
    pr->max_compute_work_group_count = L->maxComputeWorkGroupCount[0];
    pr->max_compute_work_group_size = L->maxComputeWorkGroupInvocations;
    pr->max_compute_work_group_invocations = L->maxComputeWorkGroupInvocations;
    pr->max_work_group_count_x = L->maxComputeWorkGroupCount[0];
    pr->max_work_group_count_y = L->maxComputeWorkGroupCount[1];
    pr->max_work_group_count_z = L->maxComputeWorkGroupCount[2];
    pr->max_work_group_size_x = L->maxComputeWorkGroupSize[0];
    pr->max_work_group_size_y = L->maxComputeWorkGroupSize[1];
    pr->max_work_group_size_z = L->maxComputeWorkGroupSize[2];

    pr->kfd_visible = kfd_gpu_node_count();

    detect_topology(dev);
}

/* ─── instance ───────────────────────────────────────────────────────────── */

static bool device_has_extension(VkPhysicalDevice pd, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL) != VK_SUCCESS)
        return false;
    if (count == 0) return false;
    VkExtensionProperties* ext = calloc(count, sizeof(*ext));
    if (!ext) return false;
    bool found = false;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &count, ext) == VK_SUCCESS)
        for (uint32_t i = 0; i < count; i++)
            if (strcmp(ext[i].extensionName, name) == 0) { found = true; break; }
    free(ext);
    return found;
}

static croc_device* open_device(croc_instance* inst, VkPhysicalDevice pd, uint32_t index) {
    croc_device* dev = calloc(1, sizeof(*dev));
    if (!dev) return NULL;

    dev->instance = inst;
    dev->physical_device_index = index;
    dev->physical_device = pd;

    /* physical properties, with PCI bus info when available */
    dev->has_pci_bus_info = device_has_extension(pd, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci_info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = dev->has_pci_bus_info ? &pci_info : NULL,
    };
    vkGetPhysicalDeviceProperties2(pd, &props2);
    dev->vk_props = props2.properties;
    if (dev->has_pci_bus_info) {
        dev->pci_bus = pci_info.pciBus;
        dev->pci_device = pci_info.pciDevice;
        dev->pci_function = pci_info.pciFunction;
    }

    /* find a compute-capable queue family */
    uint32_t fam_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &fam_count, NULL);
    VkQueueFamilyProperties* fams = calloc(fam_count ? fam_count : 1, sizeof(*fams));
    if (!fams) { free(dev); return NULL; }
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &fam_count, fams);
    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < fam_count; i++)
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    free(fams);
    if (qf == UINT32_MAX) {
        fprintf(stderr, "croc: device %u has no compute queue, skipped\n", index);
        free(dev);
        return NULL;
    }
    dev->queue_family_index = qf;

    /* logical device */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const char* dev_ext = dev->has_pci_bus_info ? VK_EXT_PCI_BUS_INFO_EXTENSION_NAME : NULL;
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = dev_ext ? 1u : 0u,
        .ppEnabledExtensionNames = dev_ext ? &dev_ext : NULL,
    };
    if (vkCreateDevice(pd, &dci, NULL, &dev->device) != VK_SUCCESS) {
        fprintf(stderr, "croc: vkCreateDevice failed for device %u\n", index);
        free(dev);
        return NULL;
    }
    vkGetDeviceQueue(dev->device, qf, 0, &dev->queue);

    populate_properties(dev);
    return dev;
}

croc_result croc_instance_create(croc_instance** instance) {
    if (!instance) return CROC_ERR_INVALID_ARG;

    croc_instance* inst = calloc(1, sizeof(*inst));
    if (!inst) return CROC_ERR_OUT_OF_MEMORY;

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "croc",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "croc",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    if (vkCreateInstance(&create_info, NULL, &inst->instance) != VK_SUCCESS) {
        free(inst);
        return CROC_ERR_INITIALIZATION_FAILED;
    }

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(inst->instance, &count, NULL) != VK_SUCCESS || count == 0) {
        vkDestroyInstance(inst->instance, NULL);
        free(inst);
        return CROC_ERR_NO_DEVICE;
    }
    VkPhysicalDevice* pds = calloc(count, sizeof(*pds));
    if (!pds) {
        vkDestroyInstance(inst->instance, NULL);
        free(inst);
        return CROC_ERR_OUT_OF_MEMORY;
    }
    if (vkEnumeratePhysicalDevices(inst->instance, &count, pds) != VK_SUCCESS) {
        free(pds);
        vkDestroyInstance(inst->instance, NULL);
        free(inst);
        return CROC_ERR_NO_DEVICE;
    }

    /* filter: AMD discrete GPUs only */
    for (uint32_t i = 0; i < count && inst->device_count < CROC_MAX_DEVICES; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pds[i], &props);
        if (props.vendorID != CROC_VENDOR_AMD) continue;
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;

        croc_device* dev = open_device(inst, pds[i], i);
        if (!dev) continue;

        inst->physical_devices[inst->device_count] = pds[i];
        inst->devices[inst->device_count] = dev;
        inst->device_count++;
    }
    free(pds);

    if (inst->device_count == 0) {
        vkDestroyInstance(inst->instance, NULL);
        free(inst);
        return CROC_ERR_NO_DEVICE;
    }

    *instance = inst;
    return CROC_OK;
}

void croc_instance_destroy(croc_instance* instance) {
    if (!instance) return;
    for (uint32_t i = 0; i < instance->device_count; i++) {
        croc_device* dev = instance->devices[i];
        if (!dev) continue;
        if (dev->device) vkDestroyDevice(dev->device, NULL);
        free(dev);
        instance->devices[i] = NULL;
    }
    if (instance->instance) vkDestroyInstance(instance->instance, NULL);
    free(instance);
}

uint32_t croc_instance_device_count(croc_instance* instance) {
    return instance ? instance->device_count : 0;
}

croc_result croc_instance_device_get(croc_instance* instance, uint32_t index, croc_device** device) {
    if (!instance || !device || index >= instance->device_count)
        return CROC_ERR_INVALID_ARG;
    *device = instance->devices[index];
    return CROC_OK;
}

croc_result croc_instance_device_properties(croc_device* device, croc_device_properties* props) {
    if (!device || !props) return CROC_ERR_INVALID_ARG;
    *props = device->properties;
    return CROC_OK;
}