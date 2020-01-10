#include "config.h"
#include "queue.h"

#ifdef _GLOBAL_ARRAY_QUEUE

#include <moderngpu/kernel_mergesort.hxx>

#include "util.cuh"

#include "global_array_queue.cuh"

// The global array queue needs to track if a queue item was dequeued inside the queue item,
// so that all done items are greater than those not done, therefore will be sorted to the
// end of the fel and deleted via update_fel_size_dequeue()
struct gaq_queue_item {
  queue_item item;
  bool done;
};

__host__ __device__ inline bool operator<(const gaq_queue_item& lhs, const gaq_queue_item& rhs) {
  if (lhs.done != rhs.done) {
    return lhs.done < rhs.done;
  } else {
    return lhs.item < rhs.item;
  }
}


// base queue
__device__ gaq_queue_item *fel;
#ifdef _PHOLD
__device__ long *min_ts_ptr_d;
#endif
__device__ gaq_queue_item insert_buffer[NUM_LPS * ENQUEUE_MAX];
__device__ int dequeue_count;
__device__ int dequeue_index[NUM_LPS];
__device__ int enqueue_count;
__device__ int fel_offset[NUM_LPS];
__device__ int fel_size;
__device__ int insert_count[NUM_LPS];

// pointer for mgpu::mergesort
gaq_queue_item *h_fel_ptr;

#ifdef _PHOLD
long *min_ts_ptr_h;
#endif

mgpu::standard_context_t *context;

// CUDA block size
static int num_threads_fel = min(MAX_THREADS__GLOBAL_ARRAY_QUEUE, NUM_NODES * FEL_SIZE);
static int num_blocks_fel = (NUM_NODES * FEL_SIZE + num_threads_fel - 1) / num_threads_fel;
static int num_threads_lps = min(MAX_THREADS__GLOBAL_ARRAY_QUEUE, NUM_LPS);
static int num_blocks_lps = (NUM_LPS + num_threads_lps - 1) / num_threads_lps;

// -----------------------------
// Helper functions
// -----------------------------


static void sort_fel()
{

  int h_fel_size;
  CudaSafeCall( cudaMemcpyFromSymbol(&h_fel_size, fel_size, sizeof(h_fel_size)) );

  // explicit grain size choosing
  // first parameter nt controls the size of the CTA (cooperative thread arrays, a.k.a. thread block) in threads.
  // second parameter vt is the number of work items per thread (grain size)
  typedef mgpu::launch_params_t<256, 3> launch_t;
  mgpu::mergesort<launch_t>(h_fel_ptr, h_fel_size, mgpu::less_t<gaq_queue_item>(), *context);
  // mgpu::mergesort(h_fel_ptr, h_fel_size, mgpu::less_t<gaq_queue_item>(), *context);
}

// -----------------------------
// Helper kernels
// -----------------------------

// -----------------------------
// Main kernels
// -----------------------------

static __global__ void calc_offsets()
{
  fel_offset[0] = 0;

  for (int i = 1; i < NUM_LPS; ++i) {
    fel_offset[i] = fel_offset[i - 1] + insert_count[i - 1];
  }
}

// append insert buffers to the back of the fel
static __global__ void copy_insert_buffers()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {
    for (int i = 0; i < insert_count[idx]; ++i) {

      int buf_idx = idx * ENQUEUE_MAX + i;
      int fel_idx = fel_size + fel_offset[idx] + i;

      fel[fel_idx] = insert_buffer[buf_idx];

    }
  }
}

static __global__ void reset_enqueue_count()
{
  enqueue_count = 0;
}

static __global__ void reset_dequeue_count()
{
  dequeue_count = 0;
}

static __global__ void reset_insert_count()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {

    insert_count[idx] = 0;

  }
}

static __global__ void update_fel_size_enqueue()
{
  for (int i = 0; i < NUM_LPS; ++i) {
    fel_size += insert_count[i];
  }
}

// discard dequeued items that were all sorted to the back because of their done attribute.
static __global__ void update_fel_size_dequeue()
{
  fel_size -= dequeue_count;
}


// try to find a single new, safe event for each LP and assign that to dequeue_index,
// „safe event“ as in „inside one LOOKAHEAD timeframe“.
static __global__ void mark_safe_events()
{
  // reset all dequeue_indexes
  for (int i = 0; i < NUM_LPS; ++i) {
    dequeue_index[i] = -1;
  }

  // traverse the whole fel checking for events suitable for dequeueing
  for (int i = 0; i < NUM_LPS * FEL_SIZE; ++i) {
    int lp = get_lp(fel[i].item.node);

    #ifdef _PHOLD
    // we can't assume independence of events further in the future than LOOKAHEAD, so they're not safe
    if (fel[i].item.ts >= fel[0].item.ts + LOOKAHEAD) {
      break;
    }
    #endif

    // if there's an event for every LP, the rest of the FEL can be skipped
    if (dequeue_count == NUM_LPS) {
      break;
    }

    if (dequeue_index[lp] == -1) {
      dequeue_index[lp] = i;

      // events are counted as dequeued and done here because we assume that each index we assign to
      // dequeue_index[] will be dequeued and handled later on.
      dequeue_count++;
      fel[i].done = true;
    }
  }
}

static __global__ void d_queue_init()
{
#ifdef _PHOLD
  min_ts_ptr_d = &(fel[0].item.ts);
#endif
  fel_size = 0;
}

// -----------------------------
// Queue interface -- host
// -----------------------------

void global_array_queue_init()
{
  printf("\n\n-----------------------------------\n");
  printf("[ GAQ ] Memory consumption\n");
  printf("-----------------------------------\n");
  printf(" available:     %.2f MB\n", (float) DEVICE_MEMORY_MB);
  printf(" int arrays:    %.2f KB\n", MEM_INT_ARRAYS / 1000.0);
  printf(" insert buffer: %.2f KB\n", MEM_INSERT_BUFFER / 1000.0);
  printf(" fel:           %.2f MB (%d items per FEL)\n",
      NUM_LPS * FEL_SIZE * sizeof(gaq_queue_item) / 1000000.0, FEL_SIZE);
  printf("-----------------------------------\n\n");

  // __device__ gaq_queue_item fel[NUM_NODES * FEL_SIZE];

  CudaSafeCall( cudaMalloc((void **)&h_fel_ptr, NUM_NODES * FEL_SIZE * sizeof(gaq_queue_item)) );
  CudaSafeCall( cudaMemcpyToSymbol(fel, &h_fel_ptr, sizeof(gaq_queue_item *)) );


  // cudaGetSymbolAddress((void **)&h_fel_ptr, fel);
  context = new mgpu::standard_context_t();


  d_queue_init<<<1, 1>>>();

#ifdef _PHOLD
  CudaSafeCall( cudaMemcpyFromSymbol(&min_ts_ptr_h, min_ts_ptr_d, sizeof(long *)) );
#endif
}

void global_array_queue_pre()
{
  mark_safe_events<<<1, 1>>>();
  //mark_safe_events_par<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();
}

void global_array_queue_post()
{
// START_GPU_TIMER(calc_offsets);
  calc_offsets<<<1, 1>>>();
  CudaCheckError();
// STOP_GPU_TIMER(calc_offsets);

// START_GPU_TIMER(copy_insert_buffers);
  copy_insert_buffers<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();
// STOP_GPU_TIMER(copy_insert_buffers);

// START_GPU_TIMER(update_fel_size);
  update_fel_size_enqueue<<<1, 1>>>();
  CudaCheckError();
// STOP_GPU_TIMER(update_fel_size);

// START_GPU_TIMER(sort);
  sort_fel();
// STOP_GPU_TIMER(sort);

// START_GPU_TIMER(update_fel_size);
  update_fel_size_dequeue<<<1, 1>>>();
  CudaCheckError();
// STOP_GPU_TIMER(update_fel_size);

// START_GPU_TIMER(reset_insert_count);
  reset_insert_count<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();
// STOP_GPU_TIMER(reset_insert_count);

  reset_enqueue_count<<<1, 1>>>();
  reset_dequeue_count<<<1, 1>>>();
}


void global_array_queue_finish()
{
  delete(context);
  /* printf("\n-----------------------------------\n");
  printf("[ GAQ ] Queue timers:\n");
  printf("-----------------------------------\n");
  printf("  Calc-Offsets:       %.2f ms\n", GET_SUM_MS_GPU(calc_offsets));
  printf("  Copy-Insert:        %.2f ms\n", GET_SUM_MS_GPU(copy_insert_buffers));
  printf("  Update-Size:        %.2f ms\n", GET_SUM_MS_GPU(update_fel_size));
  printf("  Reset-Insert-Count: %.2f ms\n", GET_SUM_MS_GPU(reset_insert_count));
  printf("  Sort:               %.2f ms\n", GET_SUM_MS_GPU(sort)); */
}

void global_array_queue_post_init()
{
  calc_offsets<<<1, 1>>>();
  CudaCheckError();

  copy_insert_buffers<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  update_fel_size_enqueue<<<1, 1>>>();
  CudaCheckError();

  sort_fel();

  reset_insert_count<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  reset_enqueue_count<<<1, 1>>>();
  reset_dequeue_count<<<1, 1>>>();
}

#ifdef _PHOLD
long global_array_queue_get_min_ts()
{
  long min;

  CudaSafeCall( cudaMemcpy(&min, min_ts_ptr_h, sizeof(long), cudaMemcpyDeviceToHost) );

  return min;
}
#endif

int global_array_queue_get_dequeue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, dequeue_count, sizeof(c)) );
  return c;
}

int global_array_queue_get_enqueue_count()
{
  int c = -1;
  CudaSafeCall( cudaMemcpyFromSymbol(&c, enqueue_count, sizeof(c)) );
  return c;
}

// -----------------------------
// Queue interface -- device
// -----------------------------

__device__ int global_array_queue_peek(queue_item **item, int lp)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int deq_idx = dequeue_index[idx];

  if (deq_idx != -1) {

    *item = &(fel[deq_idx]).item;
    return deq_idx;
  }
  else {
    return -1;
  }
}

__device__ void global_array_queue_set_done(int index)
{
}

__device__ bool global_array_queue_insert(queue_item item)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int lp = get_lp(item.node);

  if (insert_count[idx] >= ENQUEUE_MAX) {
    return false;
  }

  int buf_idx = idx * ENQUEUE_MAX + insert_count[idx];

  insert_buffer[buf_idx] = {.item=item, .done=false};
  insert_count[idx]++;
#ifdef _CORRECTNESS_CHECKS
  atomicAdd(&enqueue_count, 1);
#endif

  return true;
}

#endif /* #ifdef _GLOBAL_ARRAY_QUEUE */
