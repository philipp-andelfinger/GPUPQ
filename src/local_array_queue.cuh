#pragma once

#include <stdio.h>

#include "config.h"
#include "queue.cuh"
#include "util.cuh"

#define MEM_LONG_ARRAYS (1 * NUM_LPS * 4)

#ifdef _LOCAL_ARRAY_QUEUE
#define INSERT_BUFFER_SIZE 64
#define UPDATE_BUFFER_SIZE 64

#define MEM_INT_ARRAYS (4 * NUM_LPS * 2)
#define MEM_INSERT_BUFFER (1 * NUM_LPS * INSERT_BUFFER_SIZE * ITEM_BYTES)

#define MEM_BYTES ((long)1000 * 1000 * (DEVICE_MEMORY_MB))
#define MEM_FEL (MEM_BYTES - MEM_INT_ARRAYS - MEM_LONG_ARRAYS - MEM_INSERT_BUFFER)
#define FEL_SIZE (int)(MEM_BYTES / (ITEM_BYTES * NUM_NODES))

#define POPULATE_STEPS NUM_LPS
#endif

__device__ int get_min_index(int lp);


// TESTING
__global__ void d_sort(void);
