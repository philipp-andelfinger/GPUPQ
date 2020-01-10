#include <gtest/gtest.h>

#include "config.h"
#include "model.h"
#include "local_array_queue.cuh"
#include "queue.h"
#include "sort.cuh"
#include "util.cuh"

extern __device__ event fel[NUM_LPS * FEL_SIZE];
extern __device__ int fel_modified[NUM_LPS];
extern __device__ int fel_read[NUM_LPS];
extern __device__ int fel_write[NUM_LPS];

event h_fel[NUM_LPS * FEL_SIZE];
int h_fel_modified[NUM_LPS];

static __global__ void empty_kernel()
{
}

TEST (assumptions, pre)
{
  bool array_queue = false;
#ifdef _LOCAL_ARRAY_QUEUE
  array_queue = true;
#endif

  EXPECT_EQ(true, array_queue);
  EXPECT_EQ(8, FEL_SIZE);
  EXPECT_EQ(10, LOOKAHEAD);
  EXPECT_EQ(2, NUM_LPS);

#ifndef _PHOLD
  ASSERT_TRUE(false) << "the tests assume phold events stored in the queues.";
#endif
}

TEST (sort, quicksort)
{
  event test[] = {{.ts=6}, {.ts=10}, {.ts=13}, {.ts=5}, {.ts=8}, {.ts=3}, {.ts=2}};

  quicksort_seq(test, 0, 0, 6);

  EXPECT_EQ(2, test[0].ts);
  EXPECT_EQ(3, test[1].ts);
  EXPECT_EQ(5, test[2].ts);
  EXPECT_EQ(6, test[3].ts);
  EXPECT_EQ(8, test[4].ts);
  EXPECT_EQ(10, test[5].ts);
  EXPECT_EQ(13, test[6].ts);
}

static __global__ void test_sort_fel()
{
  fel[0] = {.ts=10};
  fel[1] = {.ts=6};
  fel[FEL_SIZE + 0] = {.ts=13};
  fel[FEL_SIZE + 1] = {.ts=5};

  fel_read[0] = 0;
  fel_write[0] = 2;

  fel_read[1] = 0;
  fel_write[1] = 2;

  fel_modified[0] = 1;
  fel_modified[1] = 1;
}

#ifndef _LAQ_INSERTION_SORT
TEST (sort, fel)
#else
TEST (sort, DISABLED_fel)
#endif
{
  test_sort_fel<<<1, 1>>>();

  d_sort<<<1, 2>>>();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_fel_modified, fel_modified, sizeof(h_fel_modified)) );

  EXPECT_EQ(6, h_fel[0].ts);
  EXPECT_EQ(10, h_fel[1].ts);
  EXPECT_EQ(5, h_fel[FEL_SIZE + 0].ts);
  EXPECT_EQ(13, h_fel[FEL_SIZE + 1].ts);

  EXPECT_EQ(0, h_fel_modified[0]);
  EXPECT_EQ(0, h_fel_modified[1]);
}

static __global__ void test_reduce_cub()
{
  fel[0] = {.ts=6};
  fel[1] = {.ts=10};
  fel[FEL_SIZE + 0] = {.ts=5};
  fel[FEL_SIZE + 1] = {.ts=13};

  fel_read[0] = 0;
  fel_write[0] = 2;
  fel_read[1] = 0;
  fel_write[1] = 2;
}

TEST (reduce, cub)
{
  test_reduce_cub<<<1, 1>>>();

  long min = queue_get_min_ts();

  EXPECT_EQ(5, min);
}

__device__ bool success[4];

static __global__ void test_insert_ringbuffer()
{
  success[0] = queue_insert({.node=0, .ts=1});
  success[1] = queue_insert({.node=0, .ts=2});
  success[2] = queue_insert({.node=1, .ts=3});
  success[3] = queue_insert({.node=1, .ts=4});
}

TEST (insert, ringbuffer)
{
  queue_init();

  test_insert_ringbuffer<<<1, 1>>>();

  queue_post();

  CudaSafeCall( cudaMemcpyFromSymbol(h_fel, fel, sizeof(h_fel)) );
  CudaSafeCall( cudaMemcpyFromSymbol(h_fel_modified, fel_modified, sizeof(h_fel_modified)) );

  EXPECT_EQ(1, h_fel[0].ts);
  EXPECT_EQ(2, h_fel[1].ts);
  EXPECT_EQ(3, h_fel[FEL_SIZE + 0].ts);
  EXPECT_EQ(4, h_fel[FEL_SIZE + 1].ts);

#ifndef _LAQ_INSERTION_SORT
  EXPECT_EQ(0, h_fel_modified[0]);
  EXPECT_EQ(0, h_fel_modified[1]);
#endif
}

TEST (sort, search_ringbuffer)
{
  event test[8];

  test[6] = {.ts=2};
  test[7] = {.ts=4};
  test[0] = {.ts=6};
  test[1] = {.ts=8};

  int read = 6;
  int write = 2;

  int pos_1 = search_ringbuffer(test, read, write, {.ts=1});
  int pos_5 = search_ringbuffer(test, read, write, {.ts=5});
  int pos_9 = search_ringbuffer(test, read, write, {.ts=9});

  EXPECT_EQ(5, pos_1);
  EXPECT_EQ(7, pos_5);
  EXPECT_EQ(2, pos_9);
}

TEST (sort, insert_sorted)
{
  event test[8];

  test[2] = {.ts=2};
  test[3] = {.ts=4};
  test[4] = {.ts=6};
  test[5] = {.ts=8};

  int read = 2;
  int write = 6;
  event ev = {.ts=5};

  insert_sorted(test, read, write, ev);

  EXPECT_EQ(2, test[2].ts);
  EXPECT_EQ(4, test[3].ts);
  EXPECT_EQ(5, test[4].ts);
  EXPECT_EQ(6, test[5].ts);
  EXPECT_EQ(8, test[6].ts);
}

TEST (sort, insert_sorted2)
{
  event test[8];

  test[5] = {.ts=2};
  test[6] = {.ts=4};
  test[7] = {.ts=6};
  test[0] = {.ts=8};
  test[1] = {.ts=10};

  int read = 5;
  int write = 2;
  event ev = {.ts=5};

  insert_sorted(test, read, write, ev);

  EXPECT_EQ(2, test[5].ts);
  EXPECT_EQ(4, test[6].ts);
  EXPECT_EQ(5, test[7].ts);
  EXPECT_EQ(6, test[0].ts);
  EXPECT_EQ(8, test[1].ts);
  EXPECT_EQ(10, test[2].ts);
}

TEST (sort, insert_sorted3)
{
  event test[8];

  test[5] = {.ts=2};
  test[6] = {.ts=4};
  test[7] = {.ts=6};
  test[0] = {.ts=8};
  test[1] = {.ts=10};

  int read = 5;
  int write = 2;
  event ev = {.ts=1};

  insert_sorted(test, read, write, ev);

  EXPECT_EQ(1, test[5].ts);
  EXPECT_EQ(2, test[6].ts);
  EXPECT_EQ(4, test[7].ts);
  EXPECT_EQ(6, test[0].ts);
  EXPECT_EQ(8, test[1].ts);
  EXPECT_EQ(10, test[2].ts);
}

int main(int argc, char** argv)
{
  empty_kernel<<<1, 1>>>();

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
