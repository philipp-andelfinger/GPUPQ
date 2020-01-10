#pragma once

#include "config.h"
#include "queue.cuh"
#include "util.cuh"

#ifdef _LOCAL_HEAP_QUEUE
#define INSERT_BUFFER_SIZE 20
#define MEM_INSERT_BUFFER (1 * NUM_LPS * INSERT_BUFFER_SIZE * ITEM_BYTES)
#define MEM_INT_ARRAYS (3 * NUM_LPS * 2)
#define MEM_LONG_ARRAYS (1 * NUM_LPS * 4)

#define MEM_BYTES ((long)1000 * 1000 * (DEVICE_MEMORY_MB))
#define MEM_FEL (MEM_BYTES - MEM_INT_ARRAYS - MEM_LONG_ARRAYS)
#define FEL_SIZE (MEM_BYTES / (ITEM_BYTES * NUM_NODES))

#define POPULATE_STEPS NUM_LPS
#endif

