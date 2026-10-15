/*
 * CROC — device.c
 *
 * Streams (command buffers + fences), kernel objects (descriptor/pipeline
 * management, argument binding, dispatch) and honest event stubs.
 */

#include "croc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CROC_LOCAL_SIZE_X 256u   /* must match kernelc.c execution mode */

/* ─── streams ────────────────────────────────────────────────────────────── */

croc_result croc_stream_create(croc_device* device, croc_stream** stream) {
    if (!device || !stream) return CROC_ERR_INVALID_ARG;

    croc_stream* s = calloc(1, sizeof(*s));
    if (!s) return CROC_ERR_OUT_OF_MEMORY;
    s->device = device;

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = device->queue_family_index,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(device->device, &pool_info, NULL, &s->command_pool) != VK_SUCCESS) {
        free(s);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device->device, &alloc_info, &s->command_buffer) != VK_SUCCESS) {
        vkDestroyCommandPool(device->device, s->command_pool, NULL);
        free(s);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (vkCreateFence(device->device, &fence_info, NULL, &s->fence) != VK_SUCCESS) {
        vkDestroyCommandPool(device->device, s->command_pool, NULL);
        free(s);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    *stream = s;
    return CROC_OK;
}

void croc_stream_destroy(croc_stream* stream) {
    if (!stream) return;
    VkDevice dev = stream->device->device;
    if (stream->fence) vkDestroyFence(dev, stream->fence, NULL);
    if (stream->command_pool) vkDestroyCommandPool(dev, stream->command_pool, NULL);
    free(stream);
}

croc_result croc_stream_synchronize(croc_stream* stream) {
    if (!stream) return CROC_ERR_INVALID_ARG;
    VkResult r = vkWaitForFences(stream->device->device, 1, &stream->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) return CROC_ERR_SYNC_FAILED;
    /* NOTE: do NOT reset the fence here. dispatch() pre-waits on the fence
     * before recording; resetting it in synchronize left it unsignaled with
     * no pending submission, so the SECOND dispatch on the same stream
     * deadlocked forever (found by tools/bw_test.c, which re-dispatches).
     * dispatch() resets the fence itself right before resubmitting. */
    return CROC_OK;
}

/* ─── kernels ────────────────────────────────────────────────────────────── */

static croc_result kernel_create_from_spirv(croc_device* device, const char* name,
                                            const uint32_t* spirv, size_t words,
                                            croc_kernel** kernel) {
    if (!device || !name || !kernel || !spirv || words == 0) return CROC_ERR_INVALID_ARG;

    croc_kernel* k = calloc(1, sizeof(*k));
    if (!k) return CROC_ERR_OUT_OF_MEMORY;
    k->device = device;
    k->name = strdup(name);
    k->push[1] = 0xC0FFEE00u;   /* default fill pattern */

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = words * sizeof(uint32_t),
        .pCode = spirv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(device->device, &smci, NULL, &shader);
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "croc: vkCreateShaderModule('%s') failed: %d\n", name, (int)vr);
        free((void*)k->name);
        free(k);
        return CROC_ERR_CREATE_KERNEL_FAILED;
    }

    /* descriptor set layout: one storage buffer at binding 0 */
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    if (vkCreateDescriptorSetLayout(device->device, &dslci, NULL, &k->descriptor_set_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device->device, shader, NULL);
        free((void*)k->name);
        free(k);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    /* pipeline layout: 1 set + 12-byte push constant block */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 12,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k->descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(device->device, &plci, NULL, &k->pipeline_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device->device, k->descriptor_set_layout, NULL);
        vkDestroyShaderModule(device->device, shader, NULL);
        free((void*)k->name);
        free(k);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = k->pipeline_layout,
    };
    vr = vkCreateComputePipelines(device->device, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline);
    vkDestroyShaderModule(device->device, shader, NULL);   /* pipeline holds a reference */
    if (vr != VK_SUCCESS) {
        fprintf(stderr, "croc: vkCreateComputePipelines('%s') failed: %d\n", name, (int)vr);
        vkDestroyPipelineLayout(device->device, k->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device->device, k->descriptor_set_layout, NULL);
        free((void*)k->name);
        free(k);
        return CROC_ERR_CREATE_KERNEL_FAILED;
    }

    /* descriptor pool: up to 8 sets per kernel (args can be rebound) */
    VkDescriptorPoolSize dps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 8,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 8,
        .poolSizeCount = 1,
        .pPoolSizes = &dps,
    };
    if (vkCreateDescriptorPool(device->device, &dpci, NULL, &k->descriptor_pool) != VK_SUCCESS) {
        vkDestroyPipeline(device->device, k->pipeline, NULL);
        vkDestroyPipelineLayout(device->device, k->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device->device, k->descriptor_set_layout, NULL);
        free((void*)k->name);
        free(k);
        return CROC_ERR_OUT_OF_RESOURCES;
    }

    *kernel = k;
    return CROC_OK;
}

croc_result croc_kernel_create(croc_device* device, const char* name, croc_kernel** kernel) {
    if (!device || !name || !kernel) return CROC_ERR_INVALID_ARG;

    /* offline-compiled builtin (GLSL -> SPIR-V via glslangValidator) first */
    const uint32_t* spirv = NULL;
    size_t words = 0;
    if (croc_builtin_kernel_spirv(name, &spirv, &words) == CROC_OK)
        return kernel_create_from_spirv(device, name, spirv, words, kernel);

    /* fallback: runtime SPIR-V emitter */
    uint32_t* emitted = NULL;
    croc_result r = croc_kernel_compile_spirv(name, &emitted, &words);
    if (r != CROC_OK) {
        fprintf(stderr, "croc: kernel '%s' not supported: %s\n", name, croc_result_string(r));
        return r;
    }
    r = kernel_create_from_spirv(device, name, emitted, words, kernel);
    free(emitted);
    return r;
}

croc_result croc_kernel_create_spirv(croc_device* device, const char* name,
                                     const uint32_t* spirv, size_t word_count,
                                     croc_kernel** kernel) {
    if (!device || !name || !kernel || !spirv || word_count == 0) return CROC_ERR_INVALID_ARG;
    return kernel_create_from_spirv(device, name, spirv, word_count, kernel);
}

void croc_kernel_destroy(croc_kernel* kernel) {
    if (!kernel) return;
    VkDevice dev = kernel->device->device;
    if (kernel->descriptor_pool) vkDestroyDescriptorPool(dev, kernel->descriptor_pool, NULL);
    if (kernel->pipeline) vkDestroyPipeline(dev, kernel->pipeline, NULL);
    if (kernel->pipeline_layout) vkDestroyPipelineLayout(dev, kernel->pipeline_layout, NULL);
    if (kernel->descriptor_set_layout) vkDestroyDescriptorSetLayout(dev, kernel->descriptor_set_layout, NULL);
    free((void*)kernel->name);
    free(kernel);
}

/*
 * Push constants: index is the 32-bit word slot.
 * Slot 0 (element count) and slot 2 (stride) are overwritten by the runtime
 * at dispatch time; slot 1 (pattern) is user-controlled.
 */
croc_result croc_kernel_set_arg(croc_kernel* kernel, uint32_t index, const void* data, size_t size) {
    if (!kernel || !data || size == 0) return CROC_ERR_INVALID_ARG;
    if (index * sizeof(uint32_t) + size > sizeof(kernel->push)) return CROC_ERR_SET_ARG_FAILED;
    memcpy((uint8_t*)kernel->push + index * sizeof(uint32_t), data, size);
    return CROC_OK;
}

croc_result croc_kernel_set_buffer_arg(croc_kernel* kernel, uint32_t index, croc_buffer* buffer) {
    if (!kernel || !buffer) return CROC_ERR_INVALID_ARG;
    if (index >= CROC_MAX_KERNEL_ARGS) return CROC_ERR_SET_ARG_FAILED;
    if (buffer->device != kernel->device) return CROC_ERR_SET_ARG_FAILED;
    if (kernel->args[index] != buffer) kernel->args_dirty = true;
    kernel->args[index] = buffer;
    if (index + 1 > kernel->bound_arg_count) kernel->bound_arg_count = index + 1;
    return CROC_OK;
}

croc_result croc_kernel_dispatch(croc_kernel* kernel, const uint32_t* workgroup_count, croc_stream* stream) {
    if (!kernel || !workgroup_count || !stream) return CROC_ERR_INVALID_ARG;
    if (stream->device != kernel->device) return CROC_ERR_INVALID_ARG;

    croc_buffer* arg0 = kernel->args[0];
    if (!arg0) {
        fprintf(stderr, "croc: dispatch without buffer argument 0\n");
        return CROC_ERR_SET_ARG_FAILED;
    }

    /* The fill kernel strides over gid.x only — v1 requires a 1-D grid. */
    if (workgroup_count[1] != 1 || workgroup_count[2] != 1) {
        fprintf(stderr, "croc: only 1-D dispatch grids are supported in v1\n");
        return CROC_ERR_NOT_SUPPORTED;
    }

    /* descriptor set: allocate once, rewrite when arguments change */
    if (!kernel->descriptor_set) {
        VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = kernel->descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &kernel->descriptor_set_layout,
        };
        if (vkAllocateDescriptorSets(kernel->device->device, &dsai, &kernel->descriptor_set) != VK_SUCCESS)
            return CROC_ERR_OUT_OF_RESOURCES;
        kernel->args_dirty = true;
    }
    if (kernel->args_dirty) {
        VkDescriptorBufferInfo bi = {
            .buffer = arg0->buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        VkWriteDescriptorSet w = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = kernel->descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi,
        };
        vkUpdateDescriptorSets(kernel->device->device, 1, &w, 0, NULL);
        kernel->args_dirty = false;
    }

    /* don't reset/submit over a still-pending previous submission */
    vkWaitForFences(kernel->device->device, 1, &stream->fence, VK_TRUE, UINT64_MAX);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(stream->command_buffer, &begin_info) != VK_SUCCESS)
        return CROC_ERR_LAUNCH_FAILED;

    vkCmdBindPipeline(stream->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(stream->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            kernel->pipeline_layout, 0, 1, &kernel->descriptor_set, 0, NULL);

    uint32_t push[3] = {
        (uint32_t)(arg0->size / 4),                          /* element count */
        kernel->push[1],                                     /* pattern       */
        workgroup_count[0] * CROC_LOCAL_SIZE_X,              /* stride        */
    };
    vkCmdPushConstants(stream->command_buffer, kernel->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);

    vkCmdDispatch(stream->command_buffer, workgroup_count[0], 1, 1);

    if (vkEndCommandBuffer(stream->command_buffer) != VK_SUCCESS)
        return CROC_ERR_LAUNCH_FAILED;

    if (vkResetFences(kernel->device->device, 1, &stream->fence) != VK_SUCCESS)
        return CROC_ERR_LAUNCH_FAILED;

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &stream->command_buffer,
    };
    if (vkQueueSubmit(kernel->device->queue, 1, &submit_info, stream->fence) != VK_SUCCESS)
        return CROC_ERR_DISPATCH_FAILED;

    return CROC_OK;
}

/* ─── events (not yet implemented — reported honestly) ──────────────────── */

croc_result croc_event_create(croc_event** event) {
    (void)event;
    return CROC_ERR_NOT_SUPPORTED;
}

void croc_event_destroy(croc_event* event) {
    (void)event;
}

croc_result croc_event_record(croc_event* event, croc_stream* stream) {
    (void)event; (void)stream;
    return CROC_ERR_NOT_SUPPORTED;
}

croc_result croc_event_synchronize(croc_event* event) {
    (void)event;
    return CROC_ERR_NOT_SUPPORTED;
}

croc_result croc_event_query(croc_event* event, int* completed) {
    (void)event; (void)completed;
    return CROC_ERR_NOT_SUPPORTED;
}