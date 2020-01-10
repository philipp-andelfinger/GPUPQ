#pragma once

#include "config.h"

#ifdef _LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP
#define _LOCAL_ARRAY_QUEUE
#endif

#ifdef _GLOBAL_ARRAY_QUEUE
#define _COMPILE_GLOBAL_ARRAY_QUEUE
#endif

#if defined(_HEAP_QUEUE) || defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
#define _COMPILE_HEAP_QUEUE
#endif

#if defined(_LOCAL_ARRAY_QUEUE)
#define _COMPILE_LOCAL_ARRAY_QUEUE
#endif

#if defined(_LOCAL_HEAP_QUEUE)
#define _COMPILE_LOCAL_HEAP_QUEUE
#endif

#if defined(_LOCAL_SPLAY_QUEUE)
#define _COMPILE_LOCAL_SPLAY_QUEUE
#endif

//
// Queue interface:
//

// poor man's static dispatch
#ifdef _HEAP_QUEUE
#define queue_init heap_queue_init
#define queue_post_init heap_queue_post_init
#define queue_pre heap_queue_pre
#define queue_post heap_queue_post
#define queue_get_min_ts heap_queue_get_min_ts
#define queue_finish heap_queue_finish
#define get_enqueue_count heap_queue_get_enqueue_count
#define get_dequeue_count heap_queue_get_dequeue_count
#define queue_check_phold heap_queue_check_phold
#elif defined(_GLOBAL_ARRAY_QUEUE)
#define queue_init global_array_queue_init
#define queue_post_init global_array_queue_post_init
#define queue_pre global_array_queue_pre
#define queue_post global_array_queue_post
#define queue_get_min_ts global_array_queue_get_min_ts
#define queue_finish global_array_queue_finish
#define get_enqueue_count global_array_queue_get_enqueue_count
#define get_dequeue_count global_array_queue_get_dequeue_count
#define queue_check_phold global_array_queue_check_phold
#elif defined(_LOCAL_HEAP_QUEUE)
#define queue_init local_heap_queue_init
#define queue_post_init local_heap_queue_post_init
#define queue_pre local_heap_queue_pre
#define queue_post local_heap_queue_post
#define queue_get_min_ts local_heap_queue_get_min_ts
#define queue_finish local_heap_queue_finish
#define get_enqueue_count local_heap_queue_get_enqueue_count
#define get_dequeue_count local_heap_queue_get_dequeue_count
#define queue_check_phold local_heap_queue_check_phold
#elif defined(_LOCAL_ARRAY_QUEUE)
#define queue_init local_array_queue_init
#define queue_post_init local_array_queue_post_init
#define queue_pre local_array_queue_pre
#define queue_post local_array_queue_post
#define queue_get_min_ts local_array_queue_get_min_ts
#define queue_finish local_array_queue_finish
#define get_enqueue_count local_array_queue_get_enqueue_count
#define get_dequeue_count local_array_queue_get_dequeue_count
#define queue_check_phold local_array_queue_check_phold
#elif defined(_LOCAL_SPLAY_QUEUE)
#define queue_init local_splay_queue_init
#define queue_post_init local_splay_queue_post_init
#define queue_pre local_splay_queue_pre
#define queue_post local_splay_queue_post
#define queue_get_min_ts local_splay_queue_get_min_ts
#define queue_finish local_splay_queue_finish
#define get_enqueue_count local_splay_queue_get_enqueue_count
#define get_dequeue_count local_splay_queue_get_dequeue_count
#define queue_check_phold local_splay_queue_check_phold
#endif

// initialize an empty queue with the queue parameters specified in config.h, and print some memory usage statistics
void queue_init();

// does the queue management work needed after the model inserted its initial events.
// This has to be called every time after POPULATE_STEPS events were inserted during initialization.
// POPULATE_STEPS is a constant defined in the used queue.
void queue_post_init();

// does the queue management work needed before each parallel step of model execution
void queue_pre();
// does the queue management work needed after each parallel step of model execution and correctness checks
void queue_post();

#ifdef _PHOLD
// return the key of the first item in the queue, which is the smallest key of any item
long queue_get_min_ts();
#endif

// print timing information on the queue's actions
void queue_finish();


//
// Correctness checks
//

int get_enqueue_count(void);
int get_dequeue_count(void);
// FIXME: all model specific checks should be defined in the respective model and not in the queue.
void queue_check_phold();
