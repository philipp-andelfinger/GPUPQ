#pragma once

#include "queue.cuh"
#include "model.h"
#include "util.cuh"

#ifdef _GLOBAL_ARRAY_QUEUE
#define MEM_INT_ARRAYS (3 * NUM_LPS * 2)
#define MEM_INSERT_BUFFER (1 * NUM_LPS * ENQUEUE_MAX * ITEM_BYTES)

#define MEM_BYTES (1000 * 1000 * (DEVICE_MEMORY_MB))
#define MEM_FEL (MEM_BYTES - MEM_INT_ARRAYS - MEM_INSERT_BUFFER)
#define FEL_SIZE (MIN(PHOLD_POPULATION * 2, MEM_BYTES / NUM_NODES))

#define POPULATE_STEPS NUM_LPS
#endif
