#ifndef CROC_INTERNAL_H
#define CROC_INTERNAL_H

#include "croc.h"
#include <vulkan/vulkan.h>
#include <stdbool.h>

#define CROC_MAX_DEVICES 16
#define CROC_MAX_QUEUES_PER_DEVICE 4
#define CROC_MAX_KERNEL_ARGS 8

struct croc_instance {
    VkInstance instance;
    uint32_t physical_device_count;
    VkPhysicalDevice physical_devices[CROC_MAX_DEVICES];
    croc_device* devices[CROC_MAX_DEVICES];
    uint32_t device_count;
};

struct croc_device {
    croc_instance* instance;
    uint32_t physical_device_index;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family_index;
    VkQueue queue;
    croc_device_properties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;

    /* filled by loader.c */
    VkPhysicalDeviceProperties vk_props;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
    bool has_pci_bus_info;      /* VK_EXT_pci_bus_info available on this GPU */
};

struct croc_stream {
    croc_device* device;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
};

struct croc_buffer {
    croc_device* device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    void* mapped;
};

struct croc_kernel {
    croc_device* device;
    const char* name;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;

    /* bound SSBO arguments (binding index -> buffer) */
    croc_buffer* args[CROC_MAX_KERNEL_ARGS];
    uint32_t bound_arg_count;
    bool args_dirty;            /* descriptor set must be rewritten */

    /*
     * Push constants layout for built-in kernels (8 bytes):
     *   word 0: element count (uint32) — derived from bound buffer size
     *   word 1: pattern (uint32)       — set via croc_kernel_set_arg(k, 1, ...)
     */
    uint32_t push[4];
};

struct croc_event {
    croc_stream* stream;
    VkFence fence;
    bool recorded;
};

/* kernelc.c — SPIR-V emitters (see kernelc.c for supported kernel names) */
croc_result croc_kernel_compile_spirv(const char* source, uint32_t** spirv, size_t* word_count);

/* builtin.c — offline-compiled builtin kernels (embedded by the build when
 * glslangValidator is available; returns CROC_ERR_NOT_SUPPORTED otherwise) */
croc_result croc_builtin_kernel_spirv(const char* name, const uint32_t** spirv, size_t* word_count);

#endif /* CROC_INTERNAL_H */