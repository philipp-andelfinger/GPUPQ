#include "config.h"
#include "queue.h"

#ifdef _LOCAL_HEAP_QUEUE

#include <stdio.h>
#include <cub/cub.cuh>

#include "util.cuh"
#include "assert.h"

#include "local_heap_queue.cuh"

DECLARE_GPU_TIMER(heapify);
DECLARE_GPU_TIMER(heapify_down);

// base queue
__device__ queue_item *fel;
__device__ int dequeue_count;
__device__ int enqueue_count;
__device__ int item_count[NUM_LPS];
__device__ int insert_count[NUM_LPS];
__device__ bool root_modified[NUM_LPS]; // root modified --> heapify_down
__device__ long global_min;
__device__ long lp_min_ts[NUM_LPS];

// CUDA block size
static int num_threads_lps = min(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS);
static int num_blocks_lps = (NUM_LPS + num_threads_lps - 1) / num_threads_lps;

__device__ static int d_num_threads_lps = MIN(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS);
__device__ static int d_num_blocks_lps = (NUM_LPS + MIN(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS) - 1) / MIN(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS);

// -----------------------------
// Helper functions
// -----------------------------

static inline __device__ int get_left_child_index(int node)
{
  return 2 * node + 1;
}

static inline __device__ int get_right_child_index(int node)
{
  return 2 * node + 2;
}

static inline __device__ void swap(int lp, int n1, int n2)
{
  queue_item temp = fel[lp * FEL_SIZE + n1];
  fel[lp * FEL_SIZE + n1] = fel[lp * FEL_SIZE + n2];
  fel[lp * FEL_SIZE + n2] = temp;
}

static __global__ void reset_enqueue_count()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    enqueue_count = 0;
  }
}

static __global__ void reset_dequeue_count()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    dequeue_count = 0;
  }
}

// -----------------------------
// Helper kernels
// -----------------------------

#ifdef _PHOLD
static __global__ void find_min_ts_device_pre()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {
    if (item_count[idx] > 0) {
      lp_min_ts[idx] = fel[idx * FEL_SIZE].ts;
    } else {
      lp_min_ts[idx] = LONG_MAX;
    }
  }
}
#endif

#ifdef _PHOLD
__device__ void *temp_storage = NULL;
__device__ size_t temp_storage_bytes = 0;

static __global__ void find_min_ts_device()
{
  if(!temp_storage)
  {
     cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, lp_min_ts,
         &global_min, NUM_LPS);

     CudaSafeCall( cudaMalloc(&temp_storage, temp_storage_bytes) );
  }

  cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, lp_min_ts,
      &global_min, NUM_LPS);

}


#endif

// -----------------------------
// Main kernels
// -----------------------------
__device__ void copy_last_(int idx)
{
  if (item_count[idx] > 0 && root_modified[idx]) {
    // assert(item_count[idx] > 0);
    int last_index = item_count[idx] - 1;

    swap(idx, 0, last_index);
    item_count[idx]--;
  }
}

__global__ void copy_last()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {

    if (item_count[idx] > 0 && root_modified[idx]) {
      int last_index = item_count[idx] + insert_count[idx] - 1;
      if(insert_count[idx] > 0)
        insert_count[idx]--;
      else
        item_count[idx]--;

      swap(idx, 0, last_index);
    }

  }
}

static __device__ bool heapify_node(int idx, int node)
{
  int left_child = get_left_child_index(node);
  int right_child = get_right_child_index(node);
  int min = node;

  if (left_child < item_count[idx]
      && fel[idx * FEL_SIZE + min] > fel[idx * FEL_SIZE + left_child]) {
    min = left_child;
  }
  if (right_child < item_count[idx]
      && fel[idx * FEL_SIZE + min] > fel[idx * FEL_SIZE + right_child]) {
    min = right_child;
  }
  if (min == node) {
    if (VERBOSE_DEBUG) {
      printf("  [heapify_node][LP %d] heap property restored at node %d\n", idx, node);
    }
    return false;
  }

  if (VERBOSE_DEBUG) {
#ifdef _PHOLD
      printf("  [heapify_node][LP %d] swapping node %d (=%ld) with smallest child %d (=%ld)\n",
          idx, node, fel[idx * FEL_SIZE + node].ts, min, fel[idx * FEL_SIZE + min].ts);
#endif
  }

  swap(idx, node, min);

  return true;
}


static __device__ void heapify_node_down(int idx, int node)
{
  while (true) {

    int left_child = get_left_child_index(node);
    int right_child = get_right_child_index(node);
    int min = node;

    if (left_child < item_count[idx]
        && fel[idx * FEL_SIZE + min] > fel[idx * FEL_SIZE + left_child]) {
      min = left_child;
    }
    if (right_child < item_count[idx]
        && fel[idx * FEL_SIZE + min] > fel[idx * FEL_SIZE + right_child]) {
      min = right_child;
    }
    if (min == node) {
      if (VERBOSE_DEBUG) {
        printf("  [heapify_node][LP %d] heap property restored at node %d\n", idx, node);
      }
      break;
    }

    if (VERBOSE_DEBUG) {
#ifdef _PHOLD
      printf("  [heapify_node][LP %d] swapping node %d (=%ld) with smallest child %d (=%ld)\n",
          idx, node, fel[idx * FEL_SIZE + node].ts, min, fel[idx * FEL_SIZE + min].ts);
#endif
    }

    swap(idx, node, min);
    node = min;
  }
}

static __global__ void heapify_up()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {
    int new_count = insert_count[idx];
    for(int new_nodes = 0; new_nodes < new_count; new_nodes++)
    {
      item_count[idx]++;
      insert_count[idx]--;
      for (int node = (item_count[idx] - 2) / 2;; node = (node - 1) / 2) {
        if(!heapify_node(idx, node) || node == 0)
          break;
      }
    }
  }

}



static __device__ void heapify_down_(int idx)
{
  if (root_modified[idx]) {

    root_modified[idx] = false;

    heapify_node_down(idx, 0);
  }
}



static __global__ void heapify_down()
{
  int idx = 0;

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {


    if (root_modified[idx]) {

      root_modified[idx] = false;

      heapify_node_down(idx, 0);
    }


  }
}

__global__ void local_heap_queue_print(int lp)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    printf("|");
    for (int i = 0; i < FEL_SIZE; ++i) {
      int fel_idx = lp * FEL_SIZE + i;

      if (i < item_count[lp]) {
#ifdef _PHOLD
        printf(" %ld |", fel[fel_idx].ts);
#endif
      } else {
        printf(" |");
      }

    }
    printf("\n");
  }
}

// -----------------------------
// Queue interface
// -----------------------------

void local_heap_queue_init()
{
  queue_item *h_fel;
  CudaSafeCall( cudaMalloc(&h_fel, ITEM_BYTES * FEL_SIZE * NUM_NODES) );
  CudaSafeCall( cudaMemcpyToSymbol(fel, &h_fel, sizeof(fel)) );

  printf("\n\n-----------------------------------\n");
  printf("[ LHQ ] Memory consumption\n");
  printf("-----------------------------------\n");
  printf(" available:     %.2f MB\n", (float) DEVICE_MEMORY_MB);
  printf(" int arrays:    %.2f KB\n", MEM_INT_ARRAYS / 1000.0);
  printf(" long arrays:   %.2f KB\n", MEM_LONG_ARRAYS / 1000.0);
  printf(" fel:           %.2f MB (%d items per FEL)\n",
      NUM_NODES * FEL_SIZE * sizeof(queue_item) / 1000000.0, FEL_SIZE);
  printf("-----------------------------------\n\n");
}

void local_heap_queue_finish()
{
}

int local_heap_queue_get_enqueue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, enqueue_count, sizeof(c)) );
  return c;
}

int local_heap_queue_get_dequeue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, dequeue_count, sizeof(c)) );
  return c;
}

void local_heap_queue_check_phold()
{
}

#ifdef _PHOLD
long local_heap_queue_get_min_ts()
{
  find_min_ts_device_pre<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  find_min_ts_device<<<1, 1>>>();
  CudaCheckError();

  long min;

  CudaSafeCall( cudaMemcpyFromSymbol(&min, global_min, sizeof(long)) );

  return min;
}
#endif

void local_heap_queue_pre()
{
}

__global__ void update_item_count()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {

    item_count[idx] += insert_count[idx];
    insert_count[idx] = 0;


  }
}

void local_heap_queue_post()
{
#ifdef _CORRECTNESS_CHECKS
  reset_enqueue_count<<<1, 1>>>();
  reset_dequeue_count<<<1, 1>>>();
#endif


  copy_last<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  heapify_down<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  heapify_up<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();
}


void local_heap_queue_post_init()
{
#ifdef _CORRECTNESS_CHECKS
  reset_enqueue_count<<<1, 1>>>();
  reset_dequeue_count<<<1, 1>>>();
#endif


  local_heap_queue_post();
}

__device__ bool queue_insert(queue_item item)
{

  int lp = get_lp(item.node);
  int insert_pos = item_count[lp] + atomicAdd(&insert_count[lp], 1);
  int index = lp * FEL_SIZE + insert_pos;

  if (VERBOSE_DEBUG) {
#ifdef _PHOLD
    printf("inserting item with ts %ld at insert pos %d, index %d\n", item.ts,
        insert_pos, index);
#endif
  }

  fel[index] = item;

#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&enqueue_count, 1);
#endif

  return true;
}

__device__ int local_heap_queue_remove(int lp)
{
  int index = lp * FEL_SIZE;
  queue_set_done(index);
  item_count[lp]--;

  return index;
}

__device__ int queue_peek(queue_item **item, int lp)
{
  int index = lp * FEL_SIZE;

  if (item_count[lp] > 0
#ifdef _PHOLD
      && fel[index].ts < global_min + LOOKAHEAD
#endif
      ) {

    *item = &(fel[index]);
    return index;

  }
  else {
    return -1;
  }
}

/* index should only be a multiple of FEL_SIZE (as returned by d_peek) and as thus only the root node of
 * each sub-FEL. Therefore, we can set root_modified without checking this. */
__device__ void local_heap_queue_set_done(int index)
{
  int lp = index / FEL_SIZE;
  root_modified[lp] = true;

#ifdef _A_STAR
  copy_last_(lp);

  heapify_down_(lp);

  // heapify_(lp);
#endif

#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&dequeue_count, 1);
#endif
}


__device__ bool queue_is_empty(int lp)
{
  return item_count[lp] + insert_count[lp] == 0;
}

__device__ int queue_length(int lp)
{
  return item_count[lp];
}

__device__ void queue_clear(int lp)
{
#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&dequeue_count, queue_length(lp));
#endif

  // root_modified not needed because an empty heap is already heapified.
  item_count[lp] = 0;

}

// event items from PHOLD has no id-like element for a == method
#ifdef _A_STAR
__device__ void queue_insert_or_update(queue_item item, int lp)
{
  queue_insert(item);
}

__device__ void queue_update(queue_item item, int lp)
{
  bool found = false;

  for (int i = threadIdx.x; i < queue_length(lp) && !found; i += blockDim.x) {
    queue_item *in_queue = &fel[lp * FEL_SIZE + i];

    if (*in_queue == item) {
      *in_queue = item;
      found = true;
      break;
    }

    if (found) {
      break;
    }
  }
  __syncthreads();

  assert(found);

}
#endif

#endif // #ifdef _LOCAL_HEAP_QUEUE
