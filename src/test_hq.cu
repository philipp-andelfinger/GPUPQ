#include <stdio.h>


#include <gtest/gtest.h>

#include "config.h"
#include "queue.h"
#include "queue.cuh"
#include "util.cuh"
#include "model.h"

extern __device__ bool mark_delete_update[HQ_HEAP_SIZE];
extern __device__ event dequeue_buffer[HQ_NODE_SIZE];
extern __device__ event enqueue_buffer[ENQUEUE_BUFFER_SIZE + HQ_NODE_SIZE];
extern __device__ event fel[HQ_NODE_SIZE * HQ_HEAP_SIZE];
extern __device__ int merged_mark[HQ_NODE_SIZE + ENQUEUE_BUFFER_SIZE];
extern __device__ insert_info insert_process[2 * HQ_HEAP_SIZE];
extern __device__ int current_insert_node;
extern __device__ int current_insert_process;
extern __device__ int dequeue_count;
extern __device__ int enqueue_count;
extern __device__ int event_count[HQ_HEAP_SIZE];
extern __device__ int last_node;
extern __device__ int num_delete_update;
extern __device__ int num_insert_update;
extern __device__ node_info insert_table[HQ_HEAP_SIZE];
extern __device__ event insert_merge_buffer[NUM_INSERT_PROCESSES * 2 * HQ_NODE_SIZE];

extern long current_ts;

#define TEST_INSERT_NUM 3

static __global__ void empty_kernel()
{
}

static __global__ void test_init_insert_update_pre()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};

    current_insert_node = 2;
    current_insert_process = 0;
    insert_table[2].offset = 0;

    enqueue_buffer[2 * HQ_NODE_SIZE + 0] = {.ts=30};
    enqueue_buffer[2 * HQ_NODE_SIZE + 1] = {.ts=31};
    enqueue_count = HQ_NODE_SIZE + 2;

    dequeue_count = 2;
  }
}

static __global__ void test_init_insert_update_pre_overlapping()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};

    // node 2
    fel[4] = {.ts=30};

    current_insert_node = 2;
    current_insert_process = 0;
    insert_table[2].offset = 1;

    enqueue_buffer[2 * HQ_NODE_SIZE + 0] = {.ts=30};
    enqueue_buffer[2 * HQ_NODE_SIZE + 1] = {.ts=31};
    enqueue_count = HQ_NODE_SIZE + 2;

    dequeue_count = 2;
  }
}

static __global__ void test_insert_update_merge()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[4] = {.ts=31};
    event_count[2] = 2;

    current_insert_node = 3;
    current_insert_process = 0;
    insert_table[3].offset = 0;

    enqueue_buffer[2 * HQ_NODE_SIZE + 0] = {.ts=28};
    enqueue_buffer[2 * HQ_NODE_SIZE + 1] = {.ts=31};
    enqueue_count = HQ_NODE_SIZE + 2;

    dequeue_count = 2;
  }
}

static __global__ void test_insert_update()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=31};
    event_count[2] = 2;

    current_insert_node = 3;
    current_insert_process = 0;
    insert_table[3].offset = 0;

    enqueue_buffer[2 * HQ_NODE_SIZE + 0] = {.ts=28};
    enqueue_buffer[2 * HQ_NODE_SIZE + 1] = {.ts=31};
    enqueue_count = HQ_NODE_SIZE + 2;

    dequeue_count = 2;
  }
}

static __global__ void test_insert_update2()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    event_count[2] = 1;

    current_insert_node = 2;
    current_insert_process = 0;
    insert_table[2].offset = 1;

    enqueue_buffer[2 * HQ_NODE_SIZE + 0] = {.ts=28};
    enqueue_buffer[2 * HQ_NODE_SIZE + 1] = {.ts=31};
    enqueue_count = HQ_NODE_SIZE + 2;

    dequeue_count = 2;
  }
}

static __global__ void test_insert_update_multi_insert()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=10, .node=0};
    fel[1] = {.ts=11, .node=1};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=15};
    fel[3] = {.ts=16};
    event_count[1] = 2;

    last_node = 1;
    current_insert_node = 2;
    current_insert_process = 0;

    enqueue_buffer[0] = {.ts=20};
    enqueue_buffer[1] = {.ts=21};
    enqueue_buffer[2] = {.ts=25};
    enqueue_buffer[3] = {.ts=26};
    enqueue_buffer[4] = {.ts=30};
    enqueue_buffer[5] = {.ts=31};
    enqueue_buffer[6] = {.ts=35};
    enqueue_buffer[7] = {.ts=36};

    enqueue_count = 8;
  }
}

static __global__ void test_delete_update_copy_last()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=LONG_MAX};
    fel[1] = {.ts=LONG_MAX};
    event_count[0] = 0;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=31};
    event_count[2] = 2;

    enqueue_count = 0;
    last_node = 2;
    current_insert_node = 3;

    insert_table[0].offset = 0;
    insert_table[1].offset = 2;
    insert_table[2].offset = 2;
  }
}

static __global__ void test_delete_update_copy_last_overlapping()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=LONG_MAX};
    fel[1] = {.ts=LONG_MAX};
    event_count[0] = 0;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=LONG_MAX};
    event_count[2] = 1;

    enqueue_count = 0;
    last_node = 2;
    current_insert_node = 2;
  }
}

static __global__ void test_delete_update_copy_last_nonempty_root()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=25};
    fel[1] = {.ts=LONG_MAX};
    event_count[0] = 1;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=31};
    event_count[2] = 2;

    enqueue_count = 1;
    last_node = 2;
    current_insert_node = 3;

    insert_table[0].offset = 1;
    insert_table[1].offset = 2;
    insert_table[2].offset = 2;
  }
}

static __global__ void test_min_ts()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18};
    fel[1] = {.ts=23};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=31};
    event_count[2] = 2;
  }
}

static __global__ void test_delete_update_no_insert()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=18, .node=0};
    fel[1] = {.ts=23, .node=1};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=27};
    fel[3] = {.ts=29};
    event_count[1] = 2;

    // node 2
    fel[4] = {.ts=30};
    fel[5] = {.ts=31};
    event_count[2] = 2;

    enqueue_count = 0;
    last_node = 2;
    current_insert_node = 3;

    insert_table[0].offset = 2;
    insert_table[1].offset = 2;
    insert_table[2].offset = 2;
  }
}

static __global__ void test_delete_update_not_safe()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    // node 0
    fel[0] = {.ts=5};
    fel[1] = {.ts=15};
    event_count[0] = 2;

    // node 1
    fel[2] = {.ts=18};
    fel[3] = {.ts=23};
    event_count[1] = 2;

    //enqueue_buffer[0] = {.ts=9};
    //enqueue_buffer[1] = {.ts=17};
    //enqueue_count = 2;
    last_node = 1;
    current_insert_node = 2;
  }

  if (idx == 0) {
    queue_insert({.node=0, .ts=12});
  } else if (idx == 1) {
    queue_insert({.node=0, .ts=17});
  }
}

static __global__ void test_queue_post_init_insert()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=18});
  } else if (idx == 1) {
    queue_insert({.node=0, .ts=23});
  }
}

static __global__ void test_queue_post_init_insert2()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=31});
  } else if (idx == 1) {
    queue_insert({.node=0, .ts=32});
  }
}

static __global__ void test_queue_post_init_insert3()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=25});
  } else if (idx == 1) {
    queue_insert({.node=0, .ts=27});
  }
}

static __global__ void test_queue_post_insert()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=18});
    queue_insert({.node=0, .ts=23});
  }
}

static __global__ void reset_queue()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {

    for (int i = 0; i < HQ_HEAP_SIZE; ++i) {
      event_count[i] = 0;
      mark_delete_update[i] = 0;
    }

    for (int i = 0; i < 3 * HQ_NODE_SIZE; ++i) {
      enqueue_buffer[i].ts = 0;
    }

    enqueue_count = 0;

    //current_insert_node = 1;
    //current_insert_process = 0;
    //num_insert_update = 0;
    //num_delete_update = 0;
  }
}

static __global__ void test_sort_enqueue_buffer()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    enqueue_buffer[0] = {.ts=10};
    enqueue_buffer[1] = {.ts=5};
    enqueue_buffer[2] = {.ts=20};
    enqueue_buffer[3] = {.ts=1};

    enqueue_count = 4;
  }
}

static __global__ void test_dequeue_safe()
{
  enqueue_buffer[0] = {.ts=0, .node=0};
  enqueue_buffer[1] = {.ts=1, .node=0};
  enqueue_buffer[2] = {.ts=2, .node=1};
  enqueue_buffer[3] = {.ts=3, .node=1};

  enqueue_count = 2;
  event_count[0] = 2;
}

__device__ event test_buffer[2 * HQ_NODE_SIZE];
__device__ int pos_a = -1;
__device__ int pos_b = -1;

static __global__ void test_insert_update_search_1()
{
  test_buffer[0] = {.ts=0};
  test_buffer[1] = {.ts=1};
  test_buffer[2] = {.ts=2};
  test_buffer[3] = {.ts=3};

  pos_a = search(test_buffer, 4, 2, true);
  pos_b = search(test_buffer, 4, 2, false);
}

static __global__ void test_insert_update_search_2()
{
  test_buffer[0] = {.ts=0};
  test_buffer[1] = {.ts=0};
  test_buffer[2] = {.ts=1};
  test_buffer[3] = {.ts=1};

  pos_a = search(test_buffer, 4, 0, true);
  pos_b = search(test_buffer, 4, 0, false);
}

/* Test local variables */

event h_fel[HQ_HEAP_SIZE * HQ_NODE_SIZE];
event h_dequeue_buffer[HQ_NODE_SIZE];
event h_enqueue_buffer[ENQUEUE_BUFFER_SIZE];
insert_info h_insert_process[2 * HQ_HEAP_SIZE];
int h_current_insert_node;
int h_last_node;
int h_current_insert_process;
int h_event_count[HQ_HEAP_SIZE];
int h_dequeue_count;
int h_merged_mark[HQ_NODE_SIZE + ENQUEUE_BUFFER_SIZE];
node_info h_insert_table[HQ_HEAP_SIZE];

TEST (assumptions, pre)
{
  bool heap_queue = false;
#ifdef _HEAP_QUEUE
  heap_queue = true;
#endif

  EXPECT_EQ(2, NUM_LPS);
  EXPECT_EQ(2, HQ_NODE_SIZE);
  EXPECT_EQ(true, heap_queue);
  EXPECT_EQ(10, LOOKAHEAD);
  EXPECT_GE(HQ_HEAP_SIZE, 3);

#ifndef _PHOLD
  ASSERT_TRUE(false) << "the tests assume phold events stored in the queues.";
#endif
}

TEST (insert_update, init)
{
  queue_init();

  test_init_insert_update_pre<<<1, 1>>>();

  init_insert_update(INSERT_FROM_MERGED_BUFFER);
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_insert_process, insert_process, sizeof(h_insert_process)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_process, current_insert_process, sizeof(int)) );

  EXPECT_EQ(30, h_insert_process[0].insert_buffer[0].ts);
  EXPECT_EQ(31, h_insert_process[0].insert_buffer[1].ts);
  EXPECT_EQ(2, h_insert_process[0].target_node);
  EXPECT_EQ(3, h_current_insert_node);
  EXPECT_EQ(1, h_current_insert_process);
}


TEST (insert_update, init_overlapping)
{
  queue_init();

  test_init_insert_update_pre_overlapping<<<1, 1>>>();

  init_insert_update(INSERT_FROM_MERGED_BUFFER);
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_insert_process, insert_process, sizeof(h_insert_process)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_process, current_insert_process, sizeof(int)) );

  EXPECT_EQ(30, h_insert_process[0].insert_buffer[0].ts);
  EXPECT_EQ(31, h_insert_process[1].insert_buffer[0].ts);
  EXPECT_EQ(2, h_insert_process[0].target_node);
  EXPECT_EQ(3, h_insert_process[1].target_node);
  EXPECT_EQ(3, h_current_insert_node);
  EXPECT_EQ(2, h_current_insert_process);
}

event h_insert_merge_buffer[NUM_INSERT_PROCESSES * 2 * HQ_NODE_SIZE];

TEST (insert_update, merge)
{
  queue_init();

  test_insert_update_merge<<<1, 1>>>();

  init_insert_update(INSERT_FROM_MERGED_BUFFER);
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(true);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(true);

  CudaSafeCall( cudaMemcpyFromSymbol(h_insert_merge_buffer, insert_merge_buffer,
        sizeof(h_insert_merge_buffer)) );

  EXPECT_EQ(18, h_insert_merge_buffer[0].ts);
  EXPECT_EQ(23, h_insert_merge_buffer[1].ts);
  EXPECT_EQ(28, h_insert_merge_buffer[2].ts);
  EXPECT_EQ(31, h_insert_merge_buffer[3].ts);
}


TEST (insert_update, insert_update)
{
  queue_init();

  test_insert_update<<<1, 1>>>();

  init_insert_update(INSERT_FROM_MERGED_BUFFER);
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(true);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(true);

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(false);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(false);

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(true);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(true);

  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );

  EXPECT_EQ(18, h_fel[0].ts);
  EXPECT_EQ(23, h_fel[1].ts);
  EXPECT_EQ(27, h_fel[2].ts);
  EXPECT_EQ(28, h_fel[3].ts);
  EXPECT_EQ(30, h_fel[4].ts);
  EXPECT_EQ(31, h_fel[5].ts);
  EXPECT_EQ(29, h_fel[6].ts);
  EXPECT_EQ(31, h_fel[7].ts);
}

TEST (insert_update, insert_update2)
{
  queue_init();

  test_insert_update2<<<1, 1>>>();

  init_insert_update(INSERT_FROM_MERGED_BUFFER);
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(true);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(true);

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(false);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(false);

  insert_update_pre<<<1, HQ_HEAP_SIZE>>>(true);
  insert_update();
  insert_update_post<<<1, HQ_HEAP_SIZE>>>(true);

  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_insert_table, insert_table, sizeof(h_insert_table)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );

  EXPECT_EQ(18, h_fel[0].ts);
  EXPECT_EQ(23, h_fel[1].ts);
  EXPECT_EQ(27, h_fel[2].ts);
  EXPECT_EQ(29, h_fel[3].ts);
  EXPECT_EQ(28, h_fel[4].ts);
  EXPECT_EQ(30, h_fel[5].ts);
  EXPECT_EQ(31, h_fel[6].ts);

  EXPECT_EQ(3, h_current_insert_node);
  EXPECT_EQ(1, h_insert_table[3].offset);
}

/* Insert more than 2 * N elements correctly */
TEST (insert_update_multi, preconditions)
{
  EXPECT_GE(ENQUEUE_BUFFER_SIZE, 4 * HQ_NODE_SIZE);
  EXPECT_GE(HQ_HEAP_SIZE, 6);
}

TEST (insert_update_multi, insert)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_insert_update_multi_insert<<<1, 4 * HQ_NODE_SIZE>>>();

  current_ts = 10;

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_insert_process, insert_process, sizeof(h_insert_process)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_dequeue_buffer, dequeue_buffer, sizeof(h_dequeue_buffer)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_dequeue_count, dequeue_count, sizeof(int)) );

  EXPECT_EQ(10, h_dequeue_buffer[0].ts);
  EXPECT_EQ(11, h_dequeue_buffer[1].ts);
  EXPECT_EQ(2, h_dequeue_count);

  EXPECT_EQ(15, h_fel[0].ts);
  EXPECT_EQ(16, h_fel[1].ts);
  EXPECT_EQ(20, h_fel[2].ts);
  EXPECT_EQ(21, h_fel[3].ts);
  EXPECT_EQ(25, h_fel[4].ts);
  EXPECT_EQ(26, h_fel[5].ts);
  EXPECT_EQ(30, h_fel[6].ts);
  EXPECT_EQ(31, h_fel[7].ts);

  EXPECT_EQ(35, h_insert_process[2].insert_buffer[0].ts);
  EXPECT_EQ(36, h_insert_process[2].insert_buffer[1].ts);
}

TEST (queue_post_init, insert)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_queue_post_init_insert<<<1, 2>>>();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  queue_post_init();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );

  EXPECT_EQ(18, h_fel[0].ts);
  EXPECT_EQ(23, h_fel[1].ts);

  EXPECT_EQ(2, h_event_count[0]);
}


TEST (queue_post_init, insert2)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_queue_post_init_insert<<<1, 2>>>();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  queue_post_init();

  test_queue_post_init_insert2<<<1, 2>>>();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  queue_post_init();

  test_queue_post_init_insert3<<<1, 2>>>();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  queue_post_init();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );

  EXPECT_EQ(18, h_fel[0].ts);
  EXPECT_EQ(23, h_fel[1].ts);
  EXPECT_EQ(31, h_fel[2].ts);
  EXPECT_EQ(32, h_fel[3].ts);
  EXPECT_EQ(25, h_fel[4].ts);
  EXPECT_EQ(27, h_fel[5].ts);

  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(2, h_event_count[1]);
  EXPECT_EQ(2, h_event_count[2]);
}


TEST (DISABLED_queue_post, insert)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_queue_post_insert<<<1, 1>>>();
}



TEST (delete_update, copy_last)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_delete_update_copy_last<<<1, 1>>>();

  copy_last();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_last_node, last_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );

  EXPECT_EQ(30, h_fel[0].ts);
  EXPECT_EQ(31, h_fel[1].ts);
  EXPECT_EQ(27, h_fel[2].ts);
  EXPECT_EQ(29, h_fel[3].ts);
  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(2, h_event_count[1]);
  EXPECT_EQ(0, h_event_count[2]);
  EXPECT_EQ(1, h_last_node);
  EXPECT_EQ(2, h_current_insert_node);
}

TEST (delete_update, copy_last_overlapping)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_delete_update_copy_last_overlapping<<<1, 1>>>();

  copy_last();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_last_node, last_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );

  EXPECT_EQ(29, h_fel[0].ts);
  EXPECT_EQ(30, h_fel[1].ts);
  EXPECT_EQ(27, h_fel[2].ts);
  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(1, h_event_count[1]);
  EXPECT_EQ(0, h_event_count[2]);
  EXPECT_EQ(1, h_last_node);
  EXPECT_EQ(1, h_current_insert_node);
}

TEST (delete_update, copy_last_nonempty_root)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_delete_update_copy_last_nonempty_root<<<1, 1>>>();

  copy_last();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_last_node, last_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );

  EXPECT_EQ(25, h_fel[0].ts);
  EXPECT_EQ(31, h_fel[1].ts);
  EXPECT_EQ(27, h_fel[2].ts);
  EXPECT_EQ(29, h_fel[3].ts);
  EXPECT_EQ(30, h_fel[4].ts);
  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(2, h_event_count[1]);
  EXPECT_EQ(1, h_event_count[2]);
  EXPECT_EQ(2, h_last_node);
  EXPECT_EQ(2, h_current_insert_node);
}

/*  Dequeue elements from the root without enqueueing */
TEST (delete_update, no_insert)
{
  reset_queue<<<1, 1>>>();

  queue_init();

  test_delete_update_no_insert<<<1, 1>>>();

  current_ts = 18;

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_dequeue_buffer, dequeue_buffer, sizeof(h_dequeue_buffer)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_dequeue_count, dequeue_count, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_last_node, last_node, sizeof(int)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_current_insert_node, current_insert_node, sizeof(int)) );

  EXPECT_EQ(27, h_fel[0].ts);
  EXPECT_EQ(29, h_fel[1].ts);
  EXPECT_EQ(30, h_fel[2].ts);
  EXPECT_EQ(31, h_fel[3].ts);

  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(2, h_event_count[1]);
  EXPECT_EQ(0, h_event_count[2]);

  EXPECT_EQ(1, h_last_node);
  EXPECT_EQ(2, h_current_insert_node);

  EXPECT_EQ(18, h_dequeue_buffer[0].ts);
  EXPECT_EQ(23, h_dequeue_buffer[1].ts);

  EXPECT_EQ(2, h_dequeue_count);
}

TEST (min_ts, min_ts)
{
  reset_queue<<<1, 1>>>();
  cudaDeviceSynchronize();

  queue_init();

  test_min_ts<<<1, 1>>>();
  cudaDeviceSynchronize();

  long min_ts = queue_get_min_ts();

  EXPECT_EQ(18, min_ts);
}

#ifdef _PHOLD
TEST (phold, phold_init)
{
  reset_queue<<<1, 1>>>();
  cudaDeviceSynchronize();

  queue_init();

  model_init();

  int count = 0;

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );

  for (int i = 0; i < HQ_HEAP_SIZE; i++) {
    count += h_event_count[i];
  }

  EXPECT_EQ(PHOLD_POPULATION, count);
}
#endif

/* Normal enqueue, but only half of the events at the root are safe -> only one
 * gets dequeued */
TEST (delete_update, not_safe)
{
  reset_queue<<<1, 1>>>();
  cudaDeviceSynchronize();

  queue_init();

  test_delete_update_not_safe<<<1, 2>>>();

  current_ts = 2;

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_event_count, event_count, sizeof(h_event_count)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_dequeue_buffer, dequeue_buffer, sizeof(h_dequeue_buffer)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_dequeue_count, dequeue_count, sizeof(int)) );

  EXPECT_EQ(12, h_fel[0].ts);
  EXPECT_EQ(15, h_fel[1].ts);
  EXPECT_EQ(18, h_fel[2].ts);
  EXPECT_EQ(23, h_fel[3].ts);
  EXPECT_EQ(17, h_fel[4].ts);

  EXPECT_EQ(2, h_event_count[0]);
  EXPECT_EQ(2, h_event_count[1]);
  EXPECT_EQ(1, h_event_count[2]);

  EXPECT_EQ(5, h_dequeue_buffer[0].ts);
  EXPECT_EQ(1, h_dequeue_count);
}

TEST (sort, enqueue_buffer)
{
  reset_queue<<<1, 1>>>();
  cudaDeviceSynchronize();

  queue_init();

  test_sort_enqueue_buffer<<<1, 1>>>();

  sort_enqueue_buffer();

  CudaSafeCall( cudaMemcpyFromSymbol(h_enqueue_buffer, enqueue_buffer, sizeof(h_enqueue_buffer)) );

  EXPECT_EQ(1, h_enqueue_buffer[0].ts);
  EXPECT_EQ(5, h_enqueue_buffer[1].ts);
  EXPECT_EQ(10, h_enqueue_buffer[2].ts);
  EXPECT_EQ(20, h_enqueue_buffer[3].ts);
}

/* dequeue only one safe event per lp*/
TEST (dequeue, safe)
{
  test_dequeue_safe<<<1, 1>>>();

  current_ts = 0;

  //dequeue_safe<<<1, 1>>>();
  partition_merged<<<1, 1>>>();

  CudaSafeCall( cudaMemcpyFromSymbol(h_merged_mark, merged_mark, sizeof(h_merged_mark)) );

  EXPECT_EQ(true,  h_merged_mark[0]);
  EXPECT_EQ(false, h_merged_mark[1]);
  EXPECT_EQ(true,  h_merged_mark[2]);
  EXPECT_EQ(false, h_merged_mark[3]);

  //EXPECT_EQ(0, dequeue_buffer[0].ts);
  //EXPECT_EQ(2, dequeue_buffer[1].ts);

  //EXPECT_EQ(0, dequeue_buffer[0].node);
  //EXPECT_EQ(1, dequeue_buffer[1].node);

  //EXPECT_EQ(2, dequeue_count);
}

TEST (insert_update, search1)
{
  test_insert_update_search_1<<<1, 1>>>();

  int h_pos_a;
  int h_pos_b;

  CudaSafeCall( cudaMemcpyFromSymbol(&h_pos_a, pos_a, sizeof(h_pos_a)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_pos_b, pos_b, sizeof(h_pos_b)) );

  EXPECT_EQ(3, h_pos_a);
  EXPECT_EQ(2, h_pos_b);
}

TEST (insert_update, search2)
{
  test_insert_update_search_2<<<1, 1>>>();

  int h_pos_a;
  int h_pos_b;

  CudaSafeCall( cudaMemcpyFromSymbol(&h_pos_a, pos_a, sizeof(h_pos_a)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_pos_b, pos_b, sizeof(h_pos_b)) );

  EXPECT_EQ(2, h_pos_a);
  EXPECT_EQ(0, h_pos_b);
}

int main(int argc, char** argv)
{
  empty_kernel<<<1, 1>>>();

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
