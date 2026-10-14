/* Add module parameter kfd_allow_no_atomics to kfd_device.c */
/* Insert after the existing module_param declarations, before the first function */

static bool kfd_allow_no_atomics;
module_param_named(allow_no_atomics, kfd_allow_no_atomics, bool, 0444);
MODULE_PARM_DESC(allow_no_atomics,
	"Register KFD devices even when the PCIe path rejects atomic ops");