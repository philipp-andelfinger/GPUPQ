#include "config.h"
#include "queue.h"

#ifdef _PHOLD

#include <curand_kernel.h>

#include <assert.h>
#include <random>

#include "model.h"
#include "queue.cuh"
#include "queue.h"
#include "util.cuh"

// CUDA streams
curandState *state;

// base model
__device__ int num_events[NUM_LPS] = {0};
int h_num_events[NUM_LPS];
int num_blocks_phold;
int num_threads_phold;

// -----------------------------
// Helper functions
// -----------------------------

static __device__ float rand_exponential(curandState_t *state)
{
  float ur = curand_uniform(state);
  //return 1.0;
  return logf(ur) * (-1 / PHOLD_LAMBDA);
}

// -----------------------------
// Correctness checks
// -----------------------------

__device__ long lp_last_ts[NUM_LPS];
__device__ long lp_current_ts[NUM_LPS];

static __device__ void save_lp_ts(int lp, long ts)
{
  lp_last_ts[lp] = lp_current_ts[lp];
  lp_current_ts[lp] = ts;
}

// -----------------------------
// Main kernels
// -----------------------------

static __global__ void phold_init(curandState *state, int num_states)
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < num_states;
      idx += blockDim.x * gridDim.x) {

    curand_init((SEED << 20) + idx, 0, 0, &state[idx]);
  }
}

static __global__ void phold_populate(curandState *state, int count)
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < count;
      idx += blockDim.x * gridDim.x) {

    long ts = rand_exponential(&(state[idx]));
    int node = (int) (curand_uniform(&(state[idx])) * (NUM_NODES - 1 + 0.999999));
    int lp = get_lp(node);

    /* printf("[PHOLD-populate] [B %d][T %d] Inserting event (node %d, lp %d @ ts %ld), count is %d\n",
       blockIdx.x, threadIdx.x, node, lp, ts, count); */

    bool result = queue_insert({ts, node});
    assert(result == true);
  }
}

static __global__ void handle_next_event(curandState *state)
{
#if !defined(_HEAP_QUEUE)
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {
#else
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < HQ_NODE_SIZE * HQ_NUM_QUEUES;
      idx += blockDim.x * gridDim.x) {
#endif

    event *ev = NULL;
    int event_index = queue_peek(&ev, idx);

    if (event_index != -1) {

      int lp = get_lp(ev->node);

      save_lp_ts(lp, ev->ts);
      num_events[lp]++;

      // create new event
      int node = (int) (curand_uniform(&(state[lp])) * (NUM_NODES - 1 +
            0.999999)) % NUM_NODES;
      long ts = ev->ts + LOOKAHEAD + rand_exponential(&(state[lp]));

      queue_set_done(event_index);

      /* printf("[B %d][T %d] LP %d @ ts %ld: Handling event --> LP %d @ ts %ld\n",
            blockIdx.x, threadIdx.x, get_lp(ev->node), ev->ts, get_lp(node), ts); */

      bool result = queue_insert({ts, node});
      assert(result == true /* event couldn't be inserted, this queue is full */);
    }
  }
}

// -----------------------------
// Model interface
// -----------------------------

// Fills the queue with PHOLD_POPULATION * NUM_LPS initial events
void model_init()
{
  num_threads_phold = min(MAX_THREADS__PHOLD, NUM_LPS);
  num_blocks_phold = (NUM_LPS + num_threads_phold - 1) / num_threads_phold;

  int num_states = max(POPULATE_STEPS, NUM_LPS);

/* #if HQ_NUM_QUEUES != 1
  num_states = max(num_states, HQ_NUM_QUEUES * HQ_NUM_QUEUES);
#endif */

  CudaSafeCall(
    cudaMalloc(&state,
      num_states * sizeof(curandState))
      //num_threads_phold * num_blocks_phold * sizeof(curandState))
  );

  phold_init<<<num_blocks_phold, num_threads_phold>>>(state, num_states);
  CudaCheckError();

  int remaining = PHOLD_POPULATION * NUM_LPS;

  while (remaining > 0) {
    //printf("remaining is %d\n", remaining);

    queue_pre();
    int to_insert = min(POPULATE_STEPS, remaining);
    remaining -= to_insert;

    phold_populate<<<num_blocks_phold, num_threads_phold>>>
      (state, to_insert);
    CudaCheckError();
    CudaSafeCall( cudaDeviceSynchronize() );

    queue_post_init();
    // queue_post();
    CudaCheckError();

  }
}

void model_finish()
{
}

long model_get_events()
{
  CudaSafeCall(
    cudaMemcpyFromSymbol(h_num_events, num_events, sizeof(h_num_events))
  );

  long sum_events = 0;

  for (int i = 0; i < NUM_LPS; ++i) {
    sum_events += h_num_events[i];
  }

  return sum_events;
}


void model_handle_next()
{
  handle_next_event<<<num_blocks_phold, num_threads_phold>>>
      (state);
	cudaDeviceSynchronize();
  CudaCheckError();
 // exit(1);
}


#endif // #ifdef _PHOLD
