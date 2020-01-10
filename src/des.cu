#include <chrono>

#include "config.h"
#include "model.h"
#include "queue.cuh"
#include "queue.h"
#include "util.cuh"

#include "des.h"


#ifdef COUNT_COMPARISONS
__device__ unsigned long long num_comparisons;
#endif

DECLARE_TIMER(events);
DECLARE_TIMER(initial_sort);
DECLARE_TIMER(model_init);
DECLARE_TIMER(queue_init);
DECLARE_TIMER(queue_post);
DECLARE_TIMER(queue_pre);
DECLARE_TIMER(get_min);
DECLARE_TIMER(simulation);
DECLARE_TIMER(total);

// -----------------------------
// Correctness checks
// FIXME: all model specific checks should be defined in the respective model and not in the queue.
// -----------------------------

static void check_enqueue_dequeue()
{
#if defined(_PHOLD) && !defined(LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP_ITEMS_MAX)
  int dequeue_count = get_dequeue_count();
  int enqueue_count = get_enqueue_count();

  if (dequeue_count != enqueue_count) {
    printf("[ DES ] [ PHOLD ] Error: dequeue_count (= %d) != enqueue_count (= %d)!\n",
        dequeue_count, enqueue_count);
    exit(-2);
  }
#endif
}

#ifdef _PHOLD
extern __device__ long lp_last_ts[NUM_LPS];
extern __device__ long lp_current_ts[NUM_LPS];

long h_lp_last_ts[NUM_LPS];
long h_lp_current_ts[NUM_LPS];

static void check_phold_ts()
{
  CudaSafeCall( cudaMemcpyFromSymbol(&h_lp_last_ts, lp_last_ts, sizeof(h_lp_last_ts)) );
  CudaSafeCall( cudaMemcpyFromSymbol(&h_lp_current_ts, lp_current_ts, sizeof(h_lp_current_ts)) );

  for (int i = 0; i < NUM_LPS; ++i) {
    if (h_lp_last_ts[i] >  h_lp_current_ts[i]) {
      printf("[ DES ] [ PHOLD ] Error: ts not monotonically increasing (old = %ld, current = %ld)!\n",
          h_lp_last_ts[i], h_lp_current_ts[i]);
      exit(-3);
    }
  }
}
#endif

// -----------------------------
// Main DES Functions
// -----------------------------


// initialize the priority queue that was configured in config.h
// the configured model is initialized in des_run(), though.
void des_init()
{
  cudaSetDevice(CUDA_DEVICE);

  START_TIMER(total);
  START_TIMER(queue_init);
  printf("calling queue_init()\n");
  queue_init();
  STOP_TIMER(queue_init);

  if (VERBOSE) {
    printf("[ DES ] initialization done\n");
  }
}

void des_finish()
{
  queue_finish();

  STOP_TIMER(total);

  long events = model_get_events();
  double events_per_sec = events / (GET_SUM_MS(simulation) / 1000);

#ifdef _PHOLD
    long current_ts = queue_get_min_ts();
#endif


  printf("-----------------------------------\n");
  printf("[ DES ] End of simulation\n");
#ifdef _PHOLD
  printf("        elapsed simulation time: %ld\n", current_ts);
#endif
  printf("-----------------------------------\n");
  printf("  Total time:      %.2f ms\n", GET_SUM_MS(total));
  printf("  Simulation time: %.2f ms\n", GET_SUM_MS(simulation));
  printf("  Queue init:      %.2f ms\n", GET_SUM_MS(queue_init));
  printf("  Model init:      %.2f ms\n", GET_SUM_MS(model_init));
  printf("  Queue pre:       %.2f ms (avg. %.2f µs)\n", GET_SUM_MS(queue_pre),
      GET_AVG_TIMER_US(queue_pre));
  printf("  Event handler:   %.2f ms (avg. %.2f µs)\n", GET_SUM_MS(events),
      GET_AVG_TIMER_US(events));
  printf("  Queue post:      %.2f ms (avg. %.2f µs)\n", GET_SUM_MS(queue_post),
      GET_AVG_TIMER_US(queue_post));
  printf("  Get min:         %.2f ms (avg. %.2f µs)\n", GET_SUM_MS(get_min),
      GET_AVG_TIMER_US(get_min));
  printf("\n");
  printf("  Events per sec.: %.2f = %.2e\n", events_per_sec, (float)events_per_sec);
  printf("  Events total:    %d = %.2e\n", events, (float)events);
  printf("-----------------------------------\n");

#ifdef _COUNT_COMPARISONS
  unsigned long long num_cmp;
  cudaMemcpyFromSymbol(&num_cmp, num_comparisons, sizeof(unsigned long long));
  printf("number of comparisons: %ld\n", num_cmp);
  printf("comparisons per event: %.2f\n", (float)num_cmp / events);
#endif
}

void des_run()
{
  struct timespec simulation_start, simulation_curr, last_printed;

  START_TIMER(model_init);
  model_init();
  STOP_TIMER(model_init);
  CudaSafeCall( cudaDeviceSynchronize() );

  long num_events = 0;
#ifdef _PHOLD
  long current_ts = 0;
#endif

  clock_gettime(CLOCK_MONOTONIC, &simulation_start);
  clock_gettime(CLOCK_MONOTONIC, &simulation_curr);
  clock_gettime(CLOCK_MONOTONIC, &last_printed);

  START_TIMER(simulation);
  /* int iterations = 0; */
  while (
#ifdef _PHOLD
      (MAX_TS == 0 || current_ts < MAX_TS) &&
#endif
      (MAX_EVENTS == 0 || num_events < MAX_EVENTS) &&
      (MAX_WCT == 0 || (simulation_curr.tv_sec - simulation_start.tv_sec) < MAX_WCT))
  {
#ifdef _PHOLD
    START_TIMER(get_min);
    current_ts = queue_get_min_ts();
    STOP_TIMER(get_min);
#endif

    if (VERBOSE && simulation_curr.tv_sec >= last_printed.tv_sec + PRINT_STATUS_INTERVAL) {
      last_printed = simulation_curr;

      int new_num_events = model_get_events();
#ifdef _PHOLD
      printf("[ DES ] current ts: %ld, event count: %ld (+%ld)\n", current_ts,
          new_num_events, new_num_events - num_events);
#else
      printf("[ DES ] event count: %ld (+%ld)\n",
          new_num_events, new_num_events - num_events);
#endif
      num_events = new_num_events;
    }

    START_TIMER(queue_pre);
    queue_pre();
    STOP_TIMER(queue_pre);

    CudaCheckError();

    START_TIMER(events);
    model_handle_next();
    STOP_TIMER(events);
    CudaSafeCall( cudaDeviceSynchronize() );

#ifdef _CORRECTNESS_CHECKS
    check_enqueue_dequeue();
#ifdef _PHOLD
    check_phold_ts();
#endif
#endif

    START_TIMER(queue_post);
    queue_post();
    STOP_TIMER(queue_post);

    /*if(iterations++ == 10)
      exit(1); */

#ifdef _CORRECTNESS_CHECKS
#ifdef _PHOLD
    queue_check_phold();
#endif
#endif

    clock_gettime(CLOCK_MONOTONIC, &simulation_curr);

  }
  STOP_TIMER(simulation);
  if (VERBOSE) {
    printf("[ DES ] Simulation finished, cleaning up and storing results...\n");
  }

  model_finish();
}
