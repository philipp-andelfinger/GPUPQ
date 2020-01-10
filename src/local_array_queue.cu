#include <cub/cub.cuh>
#include "config.h"
#include "queue.h"

#ifdef _COMPILE_LOCAL_ARRAY_QUEUE

#include <stdio.h>

#include "sort.cuh"
#include "util.cuh"
#include "assert.h"

#include "local_array_queue.cuh"

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
#include "heap_queue.cuh"
#endif

#include "cuda_bitset.cuh"
#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
static_assert(false, "LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS only works without _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP");
#endif

#ifdef _A_STAR
static __device__ cuda_bitset<queue_item::MAX_ID> in_queue[NUM_LPS];
#endif


static __device__ queue_item update_buffer[NUM_LPS][UPDATE_BUFFER_SIZE];
static __device__ int update_buffer_len[NUM_LPS];
#endif

/* DECLARE_GPU_TIMER(local_array_queue_pre_sort);
DECLARE_GPU_TIMER(local_array_queue_sort); */

// base queue
static __device__ queue_item *fel;
static __device__ int dequeue_count;
static __device__ int enqueue_count;
static __device__ int fel_modified[NUM_LPS];
static __device__ int fel_read[NUM_LPS];
static __device__ int fel_write[NUM_LPS];
static __device__ int fel_safe[NUM_LPS];
static __device__ int insert_count[NUM_LPS];
#ifdef _PHOLD
static __device__ long global_min;
static __device__ long lp_min_ts[NUM_LPS];
#endif
#if defined(_A_STAR) && defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
static __device__ bool lp_heap_exchange_needed[NUM_LPS];
static __device__ bool any_heap_exchange_needed;
#endif

// CUDA block size
static int num_threads_fel = min((long)MAX_THREADS__LOCAL_ARRAY_QUEUE, (long)NUM_NODES * FEL_SIZE);
static int num_blocks_fel = (NUM_NODES * FEL_SIZE + num_threads_fel - 1) / num_threads_fel;
static int num_threads_lps = min(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS);
static int num_blocks_lps = (NUM_LPS + num_threads_lps - 1) / num_threads_lps;

static int num_heap_exchanges = 0;
static int num_iterations = 0;

// -----------------------------
// Helper functions
// -----------------------------

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

static __global__ void d_init()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS * FEL_SIZE;
      idx += blockDim.x * gridDim.x) {

#ifdef _PHOLD
    if (idx == 0) {
      global_min = 0;
    }
#endif

    if (idx < NUM_LPS) {
      fel_read[idx] = 0;
      fel_write[idx] = 0;
      fel_safe[idx] = -1;

#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
      update_buffer_len[idx] = 0;
#endif
    }

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
    if (fel_read[idx] != fel_write[idx]) {
      lp_min_ts[idx] = fel[idx * FEL_SIZE + fel_read[idx]].ts;
    } else {
      lp_min_ts[idx] = LONG_MAX;
    }
  }
}

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

  // CudaSafeCall( cudaFree(temp_storage) );
}
#endif

#if defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP) && defined(_A_STAR)
static __global__ void is_heap_exchange_needed()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx == 0) {
    any_heap_exchange_needed = false;
  }
  __syncthreads();

  for (int lp = idx; lp < NUM_LPS; lp += blockDim.x * gridDim.x) {
    int r = fel_read[lp], w = fel_write[lp];
    int queue_elements = r <= w ? (w - r) : w - r + FEL_SIZE;

    if (queue_elements > 0) {
      if (queue_elements < A_STAR_MAX_PARALLEL_OL_ENTRIES) {
        lp_heap_exchange_needed[lp] = true;
        any_heap_exchange_needed = true;
      } else {
        queue_item *heap_min = NULL;
        heap_queue_peek(&heap_min, 0, lp);
        if (heap_min != NULL && fel[lp * FEL_SIZE + A_STAR_MAX_PARALLEL_OL_ENTRIES - 1] > *heap_min) {
          lp_heap_exchange_needed[lp] = true;
          any_heap_exchange_needed = true;
        } else {
          lp_heap_exchange_needed[lp] = false;
        }
      }
    } else {
      lp_heap_exchange_needed[lp] = false;
    }
  }
}

#endif

// -----------------------------
// Main kernels
// -----------------------------
static __global__ void d_insertion_sort_parallel()
{
  int idx = blockIdx.x;
  /* if(threadIdx.x == 0 && insert_count[idx] != 0)
  {

    printf("%d: before inserts\n", idx);
    // printf("\nold fel_write: %d, insert_count: %d\n", fel_write[idx], insert_count[idx]);
    queue_item prev = {-1, -1};
    for (int i = 0; (fel_read[idx] + i) % FEL_SIZE != (fel_write[idx] + insert_count[idx]) % FEL_SIZE; i++) {
      queue_item curr = fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE];


       printf("%d: %d: %d/%d %.5f %s\n", idx, i, fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE].x, fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE].y, __half2float(fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE].f), curr < prev ? " <--" : "");
      prev = curr;
    }

  } */


  /* if(!threadIdx.x)
    printf("%d: insert_count is %d\n", idx, insert_count[idx]); */
  for (int i = 0; i < insert_count[idx]; ++i) {
    queue_item item_to_insert = fel[idx * FEL_SIZE + ((fel_write[idx] + i) % FEL_SIZE)];
    /* if(!blockIdx.x && !threadIdx.x)
       printf("%d: inserting %ld from pos %d (%d)\n", idx, item_to_insert.ts, i, (fel_write[idx] + i) % FEL_SIZE); */
    __syncthreads();
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
    insert_sorted_parallel(fel + idx * FEL_SIZE, fel_read[idx], (fel_write[idx] + i) % FEL_SIZE, item_to_insert);
#else
    insert_sorted_parallel(fel + idx * FEL_SIZE, (fel_read[idx] - i + FEL_SIZE) % FEL_SIZE, fel_write[idx], item_to_insert);
#endif
    __syncthreads();

  }


  if(!threadIdx.x)
  {
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
    fel_write[idx] = (fel_write[idx] + insert_count[idx]) % FEL_SIZE;
#else
    fel_read[idx] = (fel_read[idx] - insert_count[idx] + FEL_SIZE) % FEL_SIZE;
#endif
    insert_count[idx] = 0;


  }

  /* if(!blockIdx.x && threadIdx.x == 0)
  {
    printf("%d: after all inserts, read: %d, write: %d\n", idx, fel_read[idx], fel_write[idx]);
    // printf("\nold fel_write: %d, insert_count: %d\n", fel_write[idx], insert_count[idx]);
    queue_item prev = {-1, -1};
    int num_items = 0, num_unique = 0;
    for (int i = 0; (fel_read[idx] + i) % FEL_SIZE != (fel_write[idx] + insert_count[idx]) % FEL_SIZE; i++) {
      queue_item curr = fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE];

      num_items++;
      printf("%d: %d: %d: %ld\n", idx, i,  (fel_read[idx] + i) % FEL_SIZE, fel[idx * FEL_SIZE + (fel_read[idx] + i) % FEL_SIZE].ts);
      prev = curr;
    }
  } */

}


static __global__ void d_insertion_sort()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {


#ifdef LOCAL_ARRAY_QUEUE_SINGLE_PASS
    queue_item *fel_idx = fel + idx * FEL_SIZE;
    const int read = fel_read[idx];

    int write = fel_write[idx];
    long num_insert = 0;

    // that min calculation could be done once and stored in a variable,
    // but that variable gets corrupted during the insert_sorted_single_pass call, even
    // if that function is empty. possibly a compiler bug or corruption somewhere else
    // (memcheck and // gdb find nothing of the kind)
    for(int inserted = 0; inserted < insert_count[idx];)
    {
      insert_sorted_single_pass(fel_idx, read, (write + inserted) % FEL_SIZE, min(LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE, insert_count[idx] - inserted));

      inserted += min(LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE, insert_count[idx] - inserted);
    }
#else
    for (int i = 0; i < insert_count[idx]; ++i) {
      queue_item item_to_insert = fel[idx * FEL_SIZE + ((fel_write[idx] + i) % FEL_SIZE)];

#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
      insert_sorted(fel + idx * FEL_SIZE, fel_read[idx], (fel_write[idx] + i) % FEL_SIZE, item_to_insert);
#else
      insert_sorted(fel + idx * FEL_SIZE, (fel_read[idx] - i + FEL_SIZE) % FEL_SIZE, fel_write[idx], item_to_insert);
#endif
    }
#endif


#if defined(LOCAL_ARRAY_QUEUE_BACKWARD) || defined(LOCAL_ARRAY_QUEUE_SINGLE_PASS)
    fel_write[idx] = (fel_write[idx] + insert_count[idx]) % FEL_SIZE;
#else
    fel_read[idx] = (fel_read[idx] - insert_count[idx] + FEL_SIZE) % FEL_SIZE;
#endif
    insert_count[idx] = 0;
  }

}

__global__ void d_sort()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {

    if (fel_modified[idx]) {
      // printf("calling quicksort_seq with args: %p, %d, %d, %d\n", &fel[idx * FEL_SIZE], idx, fel_read[idx], (fel_write[idx] - 1 + insert_count[idx]) % (int)FEL_SIZE)
      quicksort_seq(&fel[idx * FEL_SIZE],
          idx,
          fel_read[idx],
          (fel_write[idx] - 1 + insert_count[idx]) % (int)FEL_SIZE);

      fel_write[idx] = (fel_write[idx] + insert_count[idx]) % FEL_SIZE;
      insert_count[idx] = 0;
      fel_modified[idx] = false;
      fel_safe[idx] = fel_write[idx] - 1;
    }
  }
}

__global__ void local_array_queue_print(int lp)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx == 0) {
    int fel_min = min(fel_read[lp], fel_write[lp]);
    int fel_max = max(fel_read[lp], fel_write[lp]);

    printf("|");
    for (int i = 0; i < FEL_SIZE; ++i) {

      if (i == fel_read[lp]) {
        printf(" (r)");
      } else if (i == fel_write[lp]) {
        printf(" (w)");
      }

#ifdef _PHOLD
      int fel_idx = lp * FEL_SIZE + i;
      printf(" %ld |", fel[fel_idx].ts);
#endif
    }
    printf("\n");
  }
}

#ifdef _A_STAR
#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
static __global__ void execute_update() {
  // printf("%d: execute_update, update_buffer_len is %d\n", blockIdx.x, update_buffer_len[blockIdx.x]);
  for (int update_element = 0; update_element < update_buffer_len[blockIdx.x]; update_element++) {
    queue_item item = update_buffer[blockIdx.x][update_element];

#ifdef LOCAL_ARRAY_QUEUE_SEQUENTIAL_UPDATE
    int read = fel_read[blockIdx.x];
    int write = fel_write[blockIdx.x];
    int old_pos = find_in_ringbuffer(fel + blockIdx.x * FEL_SIZE, FEL_SIZE, read, write, item);
    assert(old_pos >= 0);
    insert_sorted(fel + blockIdx.x * FEL_SIZE, read, old_pos, item);
#else

    /* if(!threadIdx.x)
      printf("%d: updating %d/%d %.5f\n", blockIdx.x, item.x, item.y, item.f); */

    __shared__ bool found;
    __shared__ int found_pos;
    if (threadIdx.x == 0) {
      found = false;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < queue_length(blockIdx.x) + insert_count[blockIdx.x] && !found; i += blockDim.x) {
      queue_item *item_in_queue = &fel[blockIdx.x * FEL_SIZE + (fel_read[blockIdx.x] + i) % FEL_SIZE];
      // printf("%d: checking %p: %d/%d\n", blockIdx.x, item_in_queue, item_in_queue->x, item_in_queue->y);

      if (*item_in_queue == item) {
        // *item_in_queue = item;
        found = true;
        found_pos = (fel_read[blockIdx.x] + i) % FEL_SIZE;
        // printf("%d: found %d/%d %.10f at %d (%d/%d %.10f)\n", blockIdx.x, item.x, item.y, item.f, i, item_in_queue->x, item_in_queue->y, item_in_queue->f);
        break;
      }
      if(found) // no guarantee when this is visible
        break;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if(!found)
      {
        printf("%d: was looking for %d/%d (%d)\n", blockIdx.x, item.x, item.y, in_queue[blockIdx.x].get(item_id(item)));
        for (int i = threadIdx.x; i < queue_length(blockIdx.x) + insert_count[blockIdx.x] && !found; i += blockDim.x) {
          queue_item *in_queue = &fel[blockIdx.x * FEL_SIZE + (fel_read[blockIdx.x] + i) % FEL_SIZE];
          printf("%d: checking %d, %p: %d/%d %.10f\n", blockIdx.x, i, in_queue, in_queue->x, in_queue->y, in_queue->f);
        }

      }
      assert(found);
    }

    if(item < fel[blockIdx.x * FEL_SIZE + found_pos])
    {
      insert_sorted_parallel(fel + blockIdx.x * FEL_SIZE, fel_read[blockIdx.x], found_pos, item);
    }

    __syncthreads();

#endif

  }

  if (threadIdx.x == 0) {
    update_buffer_len[blockIdx.x] = 0;
  }
}
#endif
#endif


// -----------------------------
// Queue interface
// -----------------------------

void local_array_queue_init()
{
  printf("local_array_queue_init\n");
  queue_item *h_fel;
  CudaSafeCall( cudaMalloc(&h_fel, ITEM_BYTES * FEL_SIZE * NUM_NODES) );

  printf("h_fel: %p\n", h_fel);

  CudaSafeCall( cudaMemcpyToSymbol(fel, &h_fel, sizeof(fel)) );

  printf("\n\n-----------------------------------\n");
  printf("[ LAQ ] Memory consumption\n");
  printf("-----------------------------------\n");
  printf(" available:     %.5f MB\n", (float) DEVICE_MEMORY_MB);
  printf(" int arrays:    %.5f KB\n", MEM_INT_ARRAYS / 1000.0);
  printf(" long arrays:   %.5f KB\n", MEM_LONG_ARRAYS / 1000.0);
  printf(" insert buffer: %.5f KB\n", MEM_INSERT_BUFFER / 1000.0);
  printf(" fel:           %.5f MB (%d items per FEL)\n",
      NUM_LPS * FEL_SIZE * sizeof(queue_item) / 1000000.0, FEL_SIZE);
  printf("-----------------------------------\n\n");

  d_init<<<num_blocks_fel, num_threads_fel>>>();
  CudaCheckError();
  CudaSafeCall( cudaDeviceSynchronize() );

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
  heap_queue_init();
  heap_queue_post_init();
#endif
}

bool min_ts_cached = false;


#ifdef _PHOLD
long local_array_queue_get_min_ts()
{
  static long min_ts;

  if(min_ts_cached)
    return min_ts;

  find_min_ts_device_pre<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  find_min_ts_device<<<1, 1>>>();
  CudaCheckError();

  CudaSafeCall( cudaMemcpyFromSymbol(&min_ts, global_min, sizeof(long)) );

  min_ts_cached = true;

  return min_ts;
}
#endif


void local_array_queue_pre()
{
}

__device__ bool local_array_queue_insert(queue_item item)
{
  int lp = get_lp(item.node);

  int insert_pos = atomicAdd(&insert_count[lp], 1);

  if ((fel_write[lp] + insert_pos) % FEL_SIZE == fel_read[lp] && fel_read[lp] != fel_write[lp]) {
    int r = fel_read[lp], w = fel_write[lp];
    int queue_elements = r <= w ? (w - r) : w - r + FEL_SIZE;

    printf("%d is full: %d + %d items\n", lp, queue_elements, insert_pos);
    return false;
  }

  // printf("%d: placed %d/%d at %d (%d)\n", lp, item.x, item.y, insert_pos, (fel_write[lp] + insert_pos) % FEL_SIZE); return false;

  // fel[lp * FEL_SIZE + ((fel_write[lp] + insert_pos) % FEL_SIZE)] = item;
  int index = lp * FEL_SIZE + ((fel_write[lp] + insert_pos) % FEL_SIZE);
  fel[index] = item;

  fel_modified[lp] = true;

#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&enqueue_count, 1);
#endif

  return true;
}

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
// moves all elements except LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP_ITEMS_MAX from each LAQ to the / each heap
__global__ void move_to_heap()
{
  int lp = threadIdx.x + blockDim.x * blockIdx.x;

  if(lp >= NUM_LPS)
    return;
#ifdef _A_STAR
  if(!lp_heap_exchange_needed[lp])
    return;
#endif


  int r = fel_read[lp], w = fel_write[lp];
  int queue_elements = r <= w ? (w - r) : w - r + FEL_SIZE;

  while(queue_elements > LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP_ITEMS_MAX)
  {
    queue_item item = fel[lp * FEL_SIZE + ((fel_read[lp] + queue_elements - 1) % FEL_SIZE)];
    if(heap_queue_insert(item))
      queue_elements--;
    else
      break;
  }
  fel_write[lp] = (fel_read[lp] + queue_elements) % FEL_SIZE;

}

// move all elements from the / each heap's current root node to the corresponding LAQs
__global__ void move_from_heap()
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int lp = idx / HQ_NODE_SIZE;
#ifdef _A_STAR
  if(!lp_heap_exchange_needed[lp])
    return;
#endif

  queue_item *item;
  int pos = heap_queue_peek(&item, idx, lp);
  if(pos != -1)
  {
    local_array_queue_insert(*item);
  }
}
#endif

__global__ void print_element_count()
{
  int sum = 0;
  for(int lp = 0; lp < NUM_LPS; lp++)
  {
    int r = fel_read[lp], w = fel_write[lp];
    int queue_elements = r <= w ? (w - r) : w - r + FEL_SIZE;

    sum += queue_elements;
  }

  printf("%d\n", sum);
}

void local_array_queue_post()
{
#ifdef _CORRECTNESS_CHECKS
  reset_enqueue_count<<<1, 1>>>();
  reset_dequeue_count<<<1, 1>>>();
  CudaCheckError();
#endif

#ifdef _LAQ_INSERTION_SORT
#ifdef LOCAL_ARRAY_QUEUE_PARALLEL_INSERT
  d_insertion_sort_parallel<<<NUM_LPS, LOCAL_ARRAY_QUEUE_PARALLEL_INSERT_NUM_THREADS>>>();
#else
  d_insertion_sort<<<num_blocks_lps, num_threads_lps>>>();
#endif
  CudaCheckError();
#else
  d_sort<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();
#endif

#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
  execute_update<<<NUM_LPS, 1>>>();
#endif

  min_ts_cached = false;

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP

  bool exchange_needed;
#ifdef _PHOLD
  long min_ts = local_array_queue_get_min_ts();
  long heap_root_min_ts = heap_queue_root_peek_ts();
  exchange_needed = heap_root_min_ts < min_ts + LOOKAHEAD || min_ts < LOOKAHEAD;
#endif
#ifdef _A_STAR
  is_heap_exchange_needed<<<NUM_LPS, 1>>>();
  CudaCheckError();
  CudaSafeCall( cudaMemcpyFromSymbol(&exchange_needed, any_heap_exchange_needed, sizeof(exchange_needed)) );
#endif

  // exchange items between heap and local array queue if one of the conditions is met.
  // Heap exchange logic for A*:
  //  * For each LAQ: Try to have at least 4 elements in it
  //  * For each LAQ: Ensure that first A_STAR_MAX_PARALLEL_OL_ENTRIES items in the LAQ
  //    are not worse than the best element in the heap for that LAQ.
  //  * If one is not the case, add all items from the LAQ's heap's root node to the LAQ. Afterwards, both conditions
  //    are always fullfilled (if there are enough items in the heap).
  num_iterations++;
  if(exchange_needed || num_iterations < 10) // second condition is arbitrary, just to initially fill heap queue
  {
    // printf("heap exchange\n");
    num_heap_exchanges++;

    // do one move_to_heap an one move_from_heap to move the worst elements from LAQ away,
    // and get the best elements from heap
    heap_queue_pre();
    CudaCheckError();
    move_to_heap<<<num_blocks_lps, num_threads_lps>>>();
    CudaCheckError();

#ifdef _A_STAR
    // skip post of heap queue for unaffected queues, as that's still expensive if nothing changed
    bool heap_queue_post_mask[NUM_LPS];
    CudaSafeCall( cudaMemcpyFromSymbol(&heap_queue_post_mask, lp_heap_exchange_needed, sizeof(heap_queue_post_mask)) );
    set_queue_post_mask(heap_queue_post_mask);
#endif

    heap_queue_post();

    move_from_heap<<<CEILING(HQ_NODE_SIZE * HQ_NUM_QUEUES, 256), 256>>>();
    CudaCheckError();

    // for PHOLD we fill up elements for the full LOOKAHEAD, for A*, the requirements are already met.
#ifdef _PHOLD
    heap_root_min_ts = heap_queue_root_peek_ts();
    // continue to move from heap until all events from heap inside the lookahead timeframe are in the LAQ
    while(heap_root_min_ts < min_ts + LOOKAHEAD)
    {
      heap_queue_pre();
      heap_queue_post();
      move_from_heap<<<CEILING(HQ_NODE_SIZE, 256), 256>>>();
      CudaCheckError();
      heap_root_min_ts = heap_queue_root_peek_ts();
    }
#endif


    // sort again to accomodate for the elements moved from heap
#ifdef _LAQ_INSERTION_SORT
    d_insertion_sort<<<num_blocks_lps, num_threads_lps>>>();
    CudaCheckError();
#else
    d_sort<<<num_blocks_lps, num_threads_lps>>>();
    CudaCheckError();
#endif

  }
  else
  {
    // printf("not exchanging items with heap queue: %ld vs %ld\n", min_ts, heap_root_min_ts);
  }



#endif


  // print_element_count<<<1, 1>>>();
  CudaCheckError();

}

void local_array_queue_post_init()
{
  // initial sort
  local_array_queue_post();
}

int local_array_queue_get_enqueue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, enqueue_count, sizeof(c)) );
  return c;
}

int local_array_queue_get_dequeue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, dequeue_count, sizeof(c)) );
  return c;
}


__device__ void local_array_queue_set_done(int index)
{
  assert(index >= 0);
  int lp = index / FEL_SIZE;
  // printf("index is %d\n", index);

  fel_read[lp] = (fel_read[lp] + 1) % FEL_SIZE;

#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
  queue_item item = fel[index];
  // printf("%d: removed %d/%d\n", lp, item.x, item.y);
  in_queue[lp].set(item_id(item), false);
  // printf("%d: in_queue[%d] at %d is now %d\n", lp, lp, item_id(item), in_queue[lp].get(item_id(item)));
#endif

#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&dequeue_count, 1);
#endif
}

/* Removes the first non-done element in the queue whose
 * timestamp is smaller than or equal max_ts=current_ts+lookahead and returns
 * its index. Returns -1 if no such element could be found. */
__device__ int local_array_queue_remove(int lp)
{
  int index = get_min_index(lp);

  if (index != -1) {
    local_array_queue_set_done(index);
  }
  return index;
}

/* returns the index of the smallest element in the queue that is not
 * done already. */
__device__ int get_min_index(int lp)
{
  int curr = lp * FEL_SIZE + fel_read[lp];

#ifdef _PHOLD
  if (fel[curr].ts >= global_min + LOOKAHEAD) {
    return -1;
  }
#endif

  return curr;
}

/* like get_min_index, but additionally sets the item pointer to the smallest
 * item in the fel. */
__device__ int local_array_queue_peek(queue_item **item, int lp)
{
  int curr = lp * FEL_SIZE + fel_read[lp];

  if ( /* !queue_is_empty(lp) */ queue_length(lp) > 0
#ifdef _PHOLD
      && fel[curr].ts < global_min + LOOKAHEAD
#endif
#ifndef _LAQ_INSERTION_SORT
      && fel_read[lp] != (fel_safe[lp] + 1) % FEL_SIZE
#endif
    ) {

    *item = &(fel[curr]);
    return curr;
  }
  else {
    return -1;
  }
}

void local_array_queue_finish()
{
  if(num_heap_exchanges > 0) {
    printf("LAQ: average number of iterations between heap exchanges: %.5f\n", (float)num_iterations / num_heap_exchanges);
  }
  /*
  printf("\n-----------------------------------\n");
  printf("[ LAQ ] Queue timers:\n");
  printf("-----------------------------------\n");
  printf("  Pre-Sort: %.5f ms\n", GET_SUM_MS_GPU(pre_sort));
  printf("  Sort:     %.5f ms\n", GET_SUM_MS_GPU(sort));
  */
}

__device__ bool local_array_queue_is_empty(int lp)
{
  return fel_read[lp] == fel_write[lp] && !insert_count[lp];
}

__device__ int local_array_queue_length(int lp)
{
  // implements correct modulo when fel_write < fel_read
  return (fel_write[lp] - fel_read[lp] + FEL_SIZE) % FEL_SIZE;
}

__device__ void local_array_queue_clear(int lp)
{
#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&dequeue_count, queue_length(lp));
#endif

  fel_read[lp] = fel_write[lp];

#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
  in_queue[lp].clear();
  update_buffer_len[lp] = 0;
#endif

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
  heap_queue_clear(lp);
#endif
}

void local_array_queue_check_phold()
{
}

// event items from PHOLD has no id-like element for a == method
#ifdef _A_STAR
__device__ void local_array_queue_insert_or_update(queue_item item, int lp)
{
#ifdef LOCAL_ARRAY_QUEUE_UPDATE_ELEMENTS
  if (in_queue[lp].get(item_id(item))) {
    assert(update_buffer_len[lp] <= UPDATE_BUFFER_SIZE);
    int pos = atomicAdd(&update_buffer_len[lp], 1);
    // printf("%d: old update_buffer length: %d\n", lp, pos);
    update_buffer[lp][pos] = item;
  } else {
    // printf("queue_insert_or_update: %d: inserting %d/%d (%d)\n", lp, item.x, item.y, item_id(item));
    queue_insert(item);
    in_queue[lp].set(item_id(item), true);
  }
#else
  queue_insert(item);
#endif
}
#endif


#endif // #ifdef _LOCAL_ARRAY_QUEUE
