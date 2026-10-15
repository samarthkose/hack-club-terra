/*
 * gcr — Custom ROCm on Vulkan
 * gfx803 (Polaris / RX 580) Vulkan-compute-based compute runtime
 *
 * Device implementation: Vulkan compute pipeline.
 */

#include "gcr_device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Internal structures ────────────────────────────────────────────────── */

struct gcr_device {
    uint32_t gpu_index;
    gcr_device_info_t info;
    VkPhysicalDevice physical;
    VkDevice logical;
    VkQueue compute_queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkDescriptorPool desc_pool;
    VkPhysicalDeviceMemoryProperties mem_props;
};

struct gcr_buffer {
    VkBuffer vk_buf;
    VkDeviceMemory vk_mem;
    uint64_t size;
    void *mapped;
    bool host_visible;
    uint32_t memory_type;
};

struct gcr_pipeline {
    VkPipeline vk_pipe;
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    VkShaderModule shader;
};

struct gcr_stream {
    VkCommandBuffer cmd_buf;
    VkFence fence;
    gcr_device_t *dev;
    bool recording;
};

/* ─── Memory type finder ─────────────────────────────────────────────────── */

static uint32_t find_memory_type(gcr_device_t *dev, uint32_t type_bits,
                                 VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < dev->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) &&
            (dev->mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

/* ─── Device ─────────────────────────────────────────────────────────────── */

gcr_device_t *gcr_device_create(uint32_t gpu_index) {
    gcr_loader_state_t loader;
    if (gcr_loader_init(&loader) != 0) return NULL;
    if (gpu_index >= loader.device_count) {
        gcr_loader_cleanup(&loader);
        return NULL;
    }

    gcr_device_t *dev = calloc(1, sizeof(*dev));
    if (!dev) {
        gcr_loader_cleanup(&loader);
        return NULL;
    }

    dev->gpu_index = gpu_index;
    dev->info = loader.devices[gpu_index];
    dev->queue_family = dev->info.compute_queue_family;

    /* Get physical device handle — re-enumerate to get VkPhysicalDevice */
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(VK_NULL_HANDLE, &count, NULL); /* won't work — need instance */

    /* We need the instance. Let's store it from loader. For now, create a new one. */
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    };
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        free(dev);
        gcr_loader_cleanup(&loader);
        return NULL;
    }

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(inst, &dev_count, NULL);
    VkPhysicalDevice *phys = calloc(dev_count, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &dev_count, phys);

    /* Find matching physical device by device_id */
    dev->physical = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys[i], &props);
        if (props.vendorID == dev->info.vendor_id &&
            props.deviceID == dev->info.device_id) {
            /* Match by pipelineCacheUUID too if needed */
            dev->physical = phys[i];
            break;
        }
    }
    free(phys);

    if (dev->physical == VK_NULL_HANDLE) {
        vkDestroyInstance(inst, NULL);
        free(dev);
        gcr_loader_cleanup(&loader);
        return NULL;
    }

    /* Create logical device with compute queue */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = dev->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };

    if (vkCreateDevice(dev->physical, &dci, NULL, &dev->logical) != VK_SUCCESS) {
        vkDestroyInstance(inst, NULL);
        free(dev);
        gcr_loader_cleanup(&loader);
        return NULL;
    }

    vkGetDeviceQueue(dev->logical, dev->queue_family, 0, &dev->compute_queue);
    vkGetPhysicalDeviceMemoryProperties(dev->physical, &dev->mem_props);

    /* Create command pool */
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = dev->queue_family,
    };
    vkCreateCommandPool(dev->logical, &cpci, NULL, &dev->cmd_pool);

    /* Create descriptor pool */
    VkDescriptorPoolSize dps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 128,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 64,
        .poolSizeCount = 1,
        .pPoolSizes = &dps,
    };
    vkCreateDescriptorPool(dev->logical, &dpci, NULL, &dev->desc_pool);

    gcr_loader_cleanup(&loader);
    fprintf(stderr, "gcr_device: created device [%u] %s\n", gpu_index, dev->info.device_name);
    return dev;
}

void gcr_device_destroy(gcr_device_t *dev) {
    if (!dev) return;
    vkDeviceWaitIdle(dev->logical);
    if (dev->desc_pool) vkDestroyDescriptorPool(dev->logical, dev->desc_pool, NULL);
    if (dev->cmd_pool) vkDestroyCommandPool(dev->logical, dev->cmd_pool, NULL);
    vkDestroyDevice(dev->logical, NULL);
    free(dev);
}

const gcr_device_info_t *gcr_device_get_info(const gcr_device_t *dev) {
    return dev ? &dev->info : NULL;
}

/* ─── Buffers ────────────────────────────────────────────────────────────── */

gcr_buffer_t *gcr_device_alloc_buffer(gcr_device_t *dev, uint64_t size, bool host_visible) {
    if (!dev || size == 0) return NULL;

    gcr_buffer_t *buf = calloc(1, sizeof(*buf));
    if (!buf) return NULL;

    buf->size = size;
    buf->host_visible = host_visible;
    buf->mapped = NULL;

    /* Create buffer */
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(dev->logical, &bci, NULL, &buf->vk_buf) != VK_SUCCESS) {
        free(buf);
        return NULL;
    }

    /* Get memory requirements */
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(dev->logical, buf->vk_buf, &mem_req);

    /* Select memory type */
    VkMemoryPropertyFlags props = host_visible
        ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    buf->memory_type = find_memory_type(dev, mem_req.memoryTypeBits, props);
    if (buf->memory_type == UINT32_MAX) {
        vkDestroyBuffer(dev->logical, buf->vk_buf, NULL);
        free(buf);
        return NULL;
    }

    /* Allocate memory */
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = buf->memory_type,
    };
    if (vkAllocateMemory(dev->logical, &mai, NULL, &buf->vk_mem) != VK_SUCCESS) {
        vkDestroyBuffer(dev->logical, buf->vk_buf, NULL);
        free(buf);
        return NULL;
    }

    vkBindBufferMemory(dev->logical, buf->vk_buf, buf->vk_mem, 0);
    return buf;
}

void gcr_device_free_buffer(gcr_device_t *dev, gcr_buffer_t *buf) {
    if (!dev || !buf) return;
    if (buf->mapped) vkUnmapMemory(dev->logical, buf->vk_mem);
    vkDestroyBuffer(dev->logical, buf->vk_buf, NULL);
    vkFreeMemory(dev->logical, buf->vk_mem, NULL);
    free(buf);
}

uint64_t gcr_buffer_get_size(const gcr_buffer_t *buf) {
    return buf ? buf->size : 0;
}

void *gcr_buffer_map(gcr_buffer_t *buf) {
    if (!buf || buf->mapped) return buf ? buf->mapped : NULL;
    /* Need device handle to map — store it or use a different approach */
    /* For now, we'll map lazily via a stored device pointer */
    /* TODO: store dev pointer in buffer struct */
    return NULL;
}

void gcr_buffer_unmap(gcr_buffer_t *buf) {
    (void)buf; /* TODO: unmap */
}

VkBuffer gcr_buffer_get_vk(const gcr_buffer_t *buf) {
    return buf ? buf->vk_buf : VK_NULL_HANDLE;
}

/* ─── Pipelines ──────────────────────────────────────────────────────────── */

gcr_pipeline_t *gcr_device_create_pipeline(gcr_device_t *dev,
                                            const uint32_t *spirv,
                                            size_t spirv_size,
                                            const char *entry_point) {
    if (!dev || !spirv || spirv_size == 0) return NULL;

    gcr_pipeline_t *pipe = calloc(1, sizeof(*pipe));
    if (!pipe) return NULL;

    /* Create shader module */
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size,
        .pCode = spirv,
    };
    if (vkCreateShaderModule(dev->logical, &smci, NULL, &pipe->shader) != VK_SUCCESS) {
        free(pipe);
        return NULL;
    }

    /* Descriptor set layout: one storage buffer per binding */
    VkDescriptorSetLayoutBinding bindings[8] = {0};
    for (int i = 0; i < 8; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 8,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(dev->logical, &dslci, NULL, &pipe->desc_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(dev->logical, pipe->shader, NULL);
        free(pipe);
        return NULL;
    }

    /* Pipeline layout: descriptor set + push constants */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };

    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pipe->desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(dev->logical, &plci, NULL, &pipe->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(dev->logical, pipe->desc_layout, NULL);
        vkDestroyShaderModule(dev->logical, pipe->shader, NULL);
        free(pipe);
        return NULL;
    }

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = pipe->shader,
            .pName = entry_point ? entry_point : "main",
        },
        .layout = pipe->layout,
    };
    if (vkCreateComputePipelines(dev->logical, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe->vk_pipe) != VK_SUCCESS) {
        vkDestroyPipelineLayout(dev->logical, pipe->layout, NULL);
        vkDestroyDescriptorSetLayout(dev->logical, pipe->desc_layout, NULL);
        vkDestroyShaderModule(dev->logical, pipe->shader, NULL);
        free(pipe);
        return NULL;
    }

    return pipe;
}

void gcr_device_destroy_pipeline(gcr_device_t *dev, gcr_pipeline_t *pipe) {
    if (!dev || !pipe) return;
    vkDestroyPipeline(dev->logical, pipe->vk_pipe, NULL);
    vkDestroyPipelineLayout(dev->logical, pipe->layout, NULL);
    vkDestroyDescriptorSetLayout(dev->logical, pipe->desc_layout, NULL);
    vkDestroyShaderModule(dev->logical, pipe->shader, NULL);
    free(pipe);
}

/* ─── Streams (command buffers) ──────────────────────────────────────────── */

gcr_stream_t *gcr_device_create_stream(gcr_device_t *dev) {
    if (!dev) return NULL;

    gcr_stream_t *stream = calloc(1, sizeof(*stream));
    if (!stream) return NULL;

    stream->dev = dev;

    /* Allocate command buffer */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(dev->logical, &cbai, &stream->cmd_buf) != VK_SUCCESS) {
        free(stream);
        return NULL;
    }

    /* Create fence */
    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (vkCreateFence(dev->logical, &fci, NULL, &stream->fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(dev->logical, dev->cmd_pool, 1, &stream->cmd_buf);
        free(stream);
        return NULL;
    }

    return stream;
}

void gcr_device_destroy_stream(gcr_device_t *dev, gcr_stream_t *stream) {
    if (!dev || !stream) return;
    vkDestroyFence(dev->logical, stream->fence, NULL);
    vkFreeCommandBuffers(dev->logical, dev->cmd_pool, 1, &stream->cmd_buf);
    free(stream);
}

int gcr_stream_begin(gcr_stream_t *stream) {
    if (!stream || stream->recording) return -1;

    /* Reset fence and command buffer */
    vkResetFences(stream->dev->logical, 1, &stream->fence);
    vkResetCommandBuffer(stream->cmd_buf, 0);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(stream->cmd_buf, &begin_info) != VK_SUCCESS) return -1;

    stream->recording = true;
    return 0;
}

int gcr_stream_end(gcr_stream_t *stream) {
    if (!stream || !stream->recording) return -1;

    if (vkEndCommandBuffer(stream->cmd_buf) != VK_SUCCESS) return -1;
    stream->recording = false;

    /* Submit */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &stream->cmd_buf,
    };
    if (vkQueueSubmit(stream->dev->compute_queue, 1, &submit_info, stream->fence) != VK_SUCCESS)
        return -1;

    return 0;
}

int gcr_stream_dispatch(gcr_stream_t *stream,
                        gcr_pipeline_t *pipeline,
                        uint32_t x, uint32_t y, uint32_t z,
                        gcr_buffer_t **buffers,
                        uint32_t buffer_count,
                        const void *push_constants,
                        uint32_t push_size) {
    if (!stream || !pipeline || !stream->recording) return -1;

    gcr_device_t *dev = stream->dev;

    /* Allocate descriptor set */
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dev->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipeline->desc_layout,
    };
    VkDescriptorSet desc_set;
    if (vkAllocateDescriptorSets(dev->logical, &dsai, &desc_set) != VK_SUCCESS) return -1;

    /* Write descriptor sets for each buffer */
    VkDescriptorBufferInfo buf_infos[8] = {0};
    VkWriteDescriptorSet writes[8] = {0};
    uint32_t write_count = buffer_count < 8 ? buffer_count : 8;

    for (uint32_t i = 0; i < write_count; i++) {
        buf_infos[i].buffer = buffers[i]->vk_buf;
        buf_infos[i].offset = 0;
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(dev->logical, write_count, writes, 0, NULL);

    /* Bind pipeline and descriptor set */
    vkCmdBindPipeline(stream->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vk_pipe);
    vkCmdBindDescriptorSets(stream->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->layout, 0, 1, &desc_set, 0, NULL);

    /* Push constants */
    if (push_constants && push_size > 0) {
        vkCmdPushConstants(stream->cmd_buf, pipeline->layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_constants);
    }

    /* Dispatch */
    vkCmdDispatch(stream->cmd_buf, x, y, z);
    return 0;
}

int gcr_stream_wait(gcr_stream_t *stream) {
    if (!stream) return -1;
    return vkWaitForFences(stream->dev->logical, 1, &stream->fence,
                           VK_TRUE, UINT64_MAX) == VK_SUCCESS ? 0 : -1;
}

/* ─── Memcpy ─────────────────────────────────────────────────────────────── */

int gcr_device_memcpy_h2d(gcr_device_t *dev, gcr_buffer_t *buf,
                          const void *src, uint64_t offset, uint64_t size) {
    if (!dev || !buf || !src) return -1;
    if (offset + size > buf->size) return -2;

    /* For host-visible buffers, map and copy directly */
    if (buf->host_visible) {
        void *data;
        if (vkMapMemory(dev->logical, buf->vk_mem, offset, size, 0, &data) != VK_SUCCESS)
            return -3;
        memcpy(data, src, size);
        vkUnmapMemory(dev->logical, buf->vk_mem);
        return 0;
    }

    /* For device-local buffers, use staging */
    gcr_buffer_t *staging = gcr_device_alloc_buffer(dev, size, true);
    if (!staging) return -4;

    void *staging_data;
    if (vkMapMemory(dev->logical, staging->vk_mem, 0, size, 0, &staging_data) != VK_SUCCESS) {
        gcr_device_free_buffer(dev, staging);
        return -5;
    }
    memcpy(staging_data, src, size);
    vkUnmapMemory(dev->logical, staging->vk_mem);

    /* Record copy command */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev->logical, &cbai, &cmd);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = offset,
        .size = size,
    };
    vkCmdCopyBuffer(cmd, staging->vk_buf, buf->vk_buf, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(dev->logical, &fci, NULL, &fence);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(dev->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(dev->logical, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(dev->logical, fence, NULL);
    vkFreeCommandBuffers(dev->logical, dev->cmd_pool, 1, &cmd);
    gcr_device_free_buffer(dev, staging);

    return 0;
}

int gcr_device_memcpy_d2h(gcr_device_t *dev, gcr_buffer_t *buf,
                          void *dst, uint64_t offset, uint64_t size) {
    if (!dev || !buf || !dst) return -1;
    if (offset + size > buf->size) return -2;

    if (buf->host_visible) {
        void *data;
        if (vkMapMemory(dev->logical, buf->vk_mem, offset, size, 0, &data) != VK_SUCCESS)
            return -3;
        memcpy(dst, data, size);
        vkUnmapMemory(dev->logical, buf->vk_mem);
        return 0;
    }

    /* Device-local: staging buffer */
    gcr_buffer_t *staging = gcr_device_alloc_buffer(dev, size, true);
    if (!staging) return -4;

    /* Record copy */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev->logical, &cbai, &cmd);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region = {
        .srcOffset = offset,
        .dstOffset = 0,
        .size = size,
    };
    vkCmdCopyBuffer(cmd, buf->vk_buf, staging->vk_buf, 1, &copy_region);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(dev->logical, &fci, NULL, &fence);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(dev->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(dev->logical, 1, &fence, VK_TRUE, UINT64_MAX);

    /* Map staging and copy to dst */
    void *staging_data;
    if (vkMapMemory(dev->logical, staging->vk_mem, 0, size, 0, &staging_data) == VK_SUCCESS) {
        memcpy(dst, staging_data, size);
        vkUnmapMemory(dev->logical, staging->vk_mem);
    }

    vkDestroyFence(dev->logical, fence, NULL);
    vkFreeCommandBuffers(dev->logical, dev->cmd_pool, 1, &cmd);
    gcr_device_free_buffer(dev, staging);

    return 0;
}

int gcr_device_memcpy_d2d(gcr_device_t *dev,
                          gcr_buffer_t *dst, uint64_t dst_offset,
                          gcr_buffer_t *src, uint64_t src_offset,
                          uint64_t size) {
    if (!dev || !dst || !src) return -1;

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev->logical, &cbai, &cmd);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = size,
    };
    vkCmdCopyBuffer(cmd, src->vk_buf, dst->vk_buf, 1, &copy_region);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(dev->logical, &fci, NULL, &fence);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(dev->compute_queue, 1, &submit_info, fence);
    vkWaitForFences(dev->logical, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(dev->logical, fence, NULL);
    vkFreeCommandBuffers(dev->logical, dev->cmd_pool, 1, &cmd);

    return 0;
}