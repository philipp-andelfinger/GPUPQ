#pragma once

#include "config.h"
#include "queue.cuh"

#define HEAP_HEIGHT logf(HQ_HEAP_SIZE)
#define NUM_INSERT_PROCESSES (HQ_HEAP_SIZE) // we rely on that for the mgpu streams
#define ENQUEUE_BUFFER_SIZE (ENQUEUE_MAX * HQ_NODE_SIZE)

#define INSERT_FROM_ENQUEUE_BUFFER 0
#define INSERT_FROM_MERGED_BUFFER 1

#define CEILING(x,y) (((x) + (y) - 1) / (y))
#define HQ_HEAP_SIZE (CEILING(NUM_LPS, HQ_NODE_SIZE) * QUEUE_SIZE)

#ifdef _HEAP_QUEUE
#define POPULATE_STEPS HQ_NODE_SIZE

#define MEM_INT_ARRAYS 1
#define MEM_INSERT_BUFFER 1
#define FEL_SIZE 1
#endif

typedef struct {
  int offset;
  int num;
} node_info;

typedef struct {
  int current_node;
  int target_node;
  int size;
  int next_size;
  queue_item insert_buffer[HQ_NODE_SIZE];
} insert_info;

void delete_update(int qid, bool even);

#ifdef _PHOLD
__device__ int search(queue_item *item, int count, long value, bool flag_a);
#endif
__global__ void delete_update_pre(int qid, bool even);
__global__ void delete_update_post(int qid, bool even);
__global__ void merge_node_insert(int node, int proc);
__global__ void insert_update_pre(int qid, bool even);
__global__ void insert_update_pre_dummy(bool even);
__global__ void insert_update_post(int qid, bool even);
__global__ void partition_merged();
__global__ void dequeue_safe();
__device__ bool heap_queue_insert(queue_item item);
// __device__ int heap_queue_peek(queue_item **item, int pos);
__device__ int heap_queue_peek(queue_item **item, int pos, int target_qid = -1);
__device__ void heap_queue_clear(int lp);

void insert_update(int qid);
void copy_last(int qid);
void init_insert_update(int qid, int insert_source, int source_offset);
void sort_enqueue_buffer(int qid);
void set_queue_post_mask(bool *mask);

void heap_queue_pre();
void heap_queue_post();
void heap_queue_init();
void heap_queue_post_init();
long heap_queue_root_peek_ts();
