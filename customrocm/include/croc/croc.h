#ifndef CROC_H
#define CROC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct croc_instance croc_instance;
typedef struct croc_device croc_device;
typedef struct croc_stream croc_stream;
typedef struct croc_buffer croc_buffer;
typedef struct croc_kernel croc_kernel;
typedef struct croc_event croc_event;

typedef enum croc_result {
    CROC_OK = 0,
    CROC_ERR_UNKNOWN = -1,
    CROC_ERR_INVALID_ARG = -2,
    CROC_ERR_OUT_OF_MEMORY = -3,
    CROC_ERR_INITIALIZATION_FAILED = -4,
    CROC_ERR_NO_DEVICE = -5,
    CROC_ERR_NOT_SUPPORTED = -6,
    CROC_ERR_OUT_OF_RESOURCES = -7,
    CROC_ERR_LAUNCH_FAILED = -8,
    CROC_ERR_SYNC_FAILED = -9,
    CROC_ERR_MAP_FAILED = -10,
    CROC_ERR_UNMAP_FAILED = -11,
    CROC_ERR_COPY_FAILED = -12,
    CROC_ERR_CREATE_KERNEL_FAILED = -13,
    CROC_ERR_SET_ARG_FAILED = -14,
    CROC_ERR_DISPATCH_FAILED = -15
} croc_result;

typedef enum croc_device_tier {
    CROC_TIER_CPU_ATTACHED = 0,
    CROC_TIER_PCH = 1,
    CROC_TIER_UNKNOWN = 2
} croc_device_tier;

typedef struct croc_device_properties {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t subsystem_id;
    uint32_t subsystem_vendor_id;
    uint32_t gfx_target;
    uint32_t cu_count;
    uint64_t vram_size;
    uint32_t max_clock_mhz;
    uint32_t max_compute_units;
    uint32_t max_workgroup_size;
    uint32_t max_workgroup_count;
    uint32_t max_mem_alloc_size;
    uint32_t max_constant_buffer_size;
    uint32_t max_constant_args;
    uint32_t max_push_constants_size;
    uint32_t max_sampler_count;
    uint32_t max_sampled_images;
    uint32_t max_storage_buffers;
    uint32_t max_storage_images;
    uint32_t max_uniform_buffers;
    uint32_t max_input_attachments;
    uint32_t max_color_attachments;
    uint32_t max_depth_stencil_attachments;
    uint32_t max_viewports;
    uint32_t max_scissor_rects;
    uint32_t max_draw_indirect_count;
    uint32_t max_fragment_atomic_counters;
    uint32_t max_fragment_atomic_counters_buffers;
    uint32_t max_fragment_combined_atomic_counters;
    uint32_t max_fragment_combined_atomic_counters_buffers;
    uint32_t max_compute_atomic_counters;
    uint32_t max_compute_atomic_counters_buffers;
    uint32_t max_compute_shared_memory_size;
    uint32_t max_compute_work_group_count;
    uint32_t max_compute_work_group_size;
    uint32_t max_compute_work_group_invocations;
    uint32_t max_work_group_count_x;
    uint32_t max_work_group_count_y;
    uint32_t max_work_group_count_z;
    uint32_t max_work_group_size_x;
    uint32_t max_work_group_size_y;
    uint32_t max_work_group_size_z;
    uint32_t max_work_group_invocations_x;
    uint32_t max_work_group_invocations_y;
    uint32_t max_work_group_invocations_z;
    uint32_t max_work_group_invocations_w;
    uint32_t max_work_group_invocations_v;
    uint32_t max_work_group_invocations_u;
    uint32_t max_work_group_invocations_t;
    uint32_t max_work_group_invocations_s;
    uint32_t max_work_group_invocations_r;
    uint32_t max_work_group_invocations_q;
    uint32_t max_work_group_invocations_p;
    uint32_t max_work_group_invocations_o;
    uint32_t max_work_group_invocations_n;
    uint32_t max_work_group_invocations_m;
    uint32_t max_work_group_invocations_l;
    uint32_t max_work_group_invocations_k;
    uint32_t max_work_group_invocations_j;
    uint32_t max_work_group_invocations_i;
    uint32_t max_work_group_invocations_h;
    uint32_t max_work_group_invocations_g;
    uint32_t max_work_group_invocations_f;
    uint32_t max_work_group_invocations_e;
    uint32_t max_work_group_invocations_d;
    uint32_t max_work_group_invocations_c;
    uint32_t max_work_group_invocations_b;
    uint32_t max_work_group_invocations_a;
    uint32_t tier;
    uint32_t link_width;
    uint32_t link_gen;
    uint32_t kfd_visible;
} croc_device_properties;

croc_result croc_instance_create(croc_instance** instance);
void croc_instance_destroy(croc_instance* instance);
uint32_t croc_instance_device_count(croc_instance* instance);
croc_result croc_instance_device_get(croc_instance* instance, uint32_t index, croc_device** device);
croc_result croc_instance_device_properties(croc_device* device, croc_device_properties* props);

croc_result croc_stream_create(croc_device* device, croc_stream** stream);
void croc_stream_destroy(croc_stream* stream);
croc_result croc_stream_synchronize(croc_stream* stream);

croc_result croc_buffer_allocate(croc_device* device, size_t size, croc_buffer** buffer);
void croc_buffer_release(croc_buffer* buffer);
croc_result croc_buffer_map(croc_buffer* buffer, void** data);
croc_result croc_buffer_unmap(croc_buffer* buffer);

croc_result croc_kernel_create(croc_device* device, const char* name, croc_kernel** kernel);
/* create from a pre-compiled SPIR-V module (e.g. glslangValidator -V kernel.comp -o kernel.spv) */
croc_result croc_kernel_create_spirv(croc_device* device, const char* name,
                                     const uint32_t* spirv, size_t word_count,
                                     croc_kernel** kernel);
void croc_kernel_destroy(croc_kernel* kernel);
croc_result croc_kernel_set_arg(croc_kernel* kernel, uint32_t index, const void* data, size_t size);
croc_result croc_kernel_set_buffer_arg(croc_kernel* kernel, uint32_t index, croc_buffer* buffer);
croc_result croc_kernel_dispatch(croc_kernel* kernel, const uint32_t* workgroup_count, croc_stream* stream);

/* multi-GPU layer sharding (scheduler.c) */
croc_result croc_layer_plan(croc_device* const* devices, uint32_t device_count,
                            size_t total_size, uint32_t* shard_sizes);
croc_result croc_layer_dispatch(croc_kernel* const* kernels, uint32_t device_count,
                                const uint32_t* shard_sizes, croc_stream* const* streams);

croc_result croc_event_create(croc_event** event);
void croc_event_destroy(croc_event* event);
croc_result croc_event_record(croc_event* event, croc_stream* stream);
croc_result croc_event_synchronize(croc_event* event);
croc_result croc_event_query(croc_event* event, int* completed);

const char* croc_result_string(croc_result result);

#ifdef __cplusplus
}
#endif

#endif /* CROC_H */