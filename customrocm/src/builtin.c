/*
 * CROC — builtin.c
 *
 * Offline-compiled builtin kernels. The build embeds SPIR-V produced by
 * glslangValidator from shaders/ (e.g. shaders/fill.comp) into generated headers:
 *
 *   shaders/fill.comp  --(glslangValidator -V)-->  build/fill.spv
 *        --(scripts/spv_to_header.py)-->  build/fill_spv.h  (fill_spv[])
 *
 * Compiled with -DCROC_HAVE_BUILTIN_SPIRV -I<build> when glslangValidator is
 * available; without it, this file exposes no builtins and the runtime falls
 * back to the hand-rolled emitter in kernelc.c.
 */

#include "croc_internal.h"

#include <string.h>

#ifdef CROC_HAVE_BUILTIN_SPIRV
#include "fill_spv.h"
#endif

croc_result croc_builtin_kernel_spirv(const char* name, const uint32_t** spirv, size_t* word_count) {
#ifndef CROC_HAVE_BUILTIN_SPIRV
    (void)name; (void)spirv; (void)word_count;
    return CROC_ERR_NOT_SUPPORTED;
#else
    if (!name || !spirv || !word_count) return CROC_ERR_INVALID_ARG;

    if (strcmp(name, "fill") == 0 || strcmp(name, "main") == 0) {
        *spirv = fill_spv;
        *word_count = sizeof(fill_spv) / sizeof(fill_spv[0]);
        return CROC_OK;
    }

    return CROC_ERR_NOT_SUPPORTED;
#endif
}