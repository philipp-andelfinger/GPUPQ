#pragma once

#include "config.h"
#include "queue.cuh"
#include "util.cuh"

#ifdef _LOCAL_SPLAY_QUEUE

struct tree_node {
  tree_node *left, *right;
  queue_item item;
};

#define MEM_BYTES ((long)1000 * 1000 * (DEVICE_MEMORY_MB))
#define FEL_SIZE (MEM_BYTES / (ITEM_BYTES * NUM_NODES) / 8)
#define MALLOC_BUF_SIZE (MEM_BYTES / (sizeof(tree_node) * NUM_NODES))
#define POPULATE_STEPS NUM_LPS
#endif
