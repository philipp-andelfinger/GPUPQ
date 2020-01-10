#include <gtest/gtest.h>

#include "config.h"
#include "model.h"
#include "global_array_queue.cuh"
#include "queue.h"
#include "sort.cuh"
#include "util.cuh"

extern __device__ event fel[NUM_LPS * FEL_SIZE];
extern __device__ int fel_size;

event h_fel[NUM_LPS * FEL_SIZE];

static __global__ void empty_kernel()
{
}

TEST (pre, assumptions)
{
  bool gaq_set = false;
#ifdef _GLOBAL_ARRAY_QUEUE
  gaq_set = true;
#endif

  int num_lps = NUM_LPS;
  int fel_size = FEL_SIZE;
  int enqueue_max = ENQUEUE_MAX;

  ASSERT_EQ(true, gaq_set);
  ASSERT_EQ(2, num_lps);
  ASSERT_EQ(5, fel_size);
  ASSERT_EQ(4, enqueue_max);

#ifndef _PHOLD
  ASSERT_TRUE(false) << "the tests assume phold events stored in the queues.";
#endif
}


__global__ void test_insert_init()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=1});
    queue_insert({.node=1, .ts=2});
  }
  if (idx == 1) {
    queue_insert({.node=0, .ts=3});
    queue_insert({.node=1, .ts=4});
  }
}

TEST (insert, init)
{
  queue_init();

  test_insert_init<<<1, 2>>>();
  CudaCheckError();

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );

  EXPECT_EQ(1, h_fel[0].ts);
  EXPECT_EQ(2, h_fel[1].ts);
  EXPECT_EQ(3, h_fel[2].ts);
  EXPECT_EQ(4, h_fel[3].ts);
}

__global__ void test_insert_cont()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    queue_insert({.node=0, .ts=3});
  }
  if (idx == 1) {
    queue_insert({.node=1, .ts=5});
  }
}

TEST (insert, cont)
{
  queue_init();

  test_insert_init<<<1, 2>>>();
  CudaCheckError();

  queue_post();

  test_insert_cont<<<1, 2>>>();
  CudaCheckError();

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );

  EXPECT_EQ(1, h_fel[0].ts);
  EXPECT_EQ(2, h_fel[1].ts);
  EXPECT_EQ(3, h_fel[2].ts);
  EXPECT_EQ(3, h_fel[3].ts);
  EXPECT_EQ(4, h_fel[4].ts);
  EXPECT_EQ(5, h_fel[5].ts);
}

__global__ void test_dequeue_dequeue_pre()
{
  fel[0] = {.ts=1, .node=0};
  fel[1] = {.ts=2, .node=0};
  fel[2] = {.ts=3, .node=1};
  fel[3] = {.ts=4, .node=1};

  fel_size = 4;
}

__device__ event test_dequeue_dequeue_arr[2];

__global__ void test_dequeue_dequeue()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  event *ev;

  queue_peek(&ev, -1);

  test_dequeue_dequeue_arr[idx] = *ev;
}

TEST (dequeue, dequeue)
{
  queue_init();

  test_dequeue_dequeue_pre<<<1, 1>>>();
  CudaCheckError();

  queue_post();

  queue_pre();

  test_dequeue_dequeue<<<1, 2>>>();

  event h_test_dequeue_dequeue_arr[2];

  CudaSafeCall( cudaMemcpyFromSymbol(h_test_dequeue_dequeue_arr,
        test_dequeue_dequeue_arr, sizeof(h_test_dequeue_dequeue_arr)) );

  EXPECT_EQ(1, h_test_dequeue_dequeue_arr[0].ts);
  EXPECT_EQ(3, h_test_dequeue_dequeue_arr[1].ts);
}

int main(int argc, char** argv)
{
  empty_kernel<<<1, 1>>>();

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
