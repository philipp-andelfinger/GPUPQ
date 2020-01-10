#pragma once

#include "config.h"
#include "queue.h"
#include "queue_item.h"

#ifdef _HEAP_QUEUE
#define queue_insert heap_queue_insert
#define queue_peek heap_queue_peek
#define queue_set_done heap_queue_set_done
#define queue_print heap_queue_print
#define queue_length heap_queue_length
#define queue_is_empty heap_queue_is_empty
#define queue_insert_or_update heap_queue_insert_or_update
#define queue_clear heap_queue_clear
#elif defined(_GLOBAL_ARRAY_QUEUE)
#define queue_insert global_array_queue_insert
#define queue_peek global_array_queue_peek
#define queue_set_done global_array_queue_set_done
#define queue_print global_array_queue_print
#elif defined(_LOCAL_HEAP_QUEUE)
#define queue_insert local_heap_queue_insert
#define queue_peek local_heap_queue_peek
#define queue_set_done local_heap_queue_set_done
#define queue_print local_heap_queue_print
#elif defined(_LOCAL_ARRAY_QUEUE)
#define queue_insert local_array_queue_insert
#define queue_peek local_array_queue_peek
#define queue_set_done local_array_queue_set_done
#define queue_print local_array_queue_print
#define queue_length local_array_queue_length
#define queue_is_empty local_array_queue_is_empty
#define queue_insert_or_update local_array_queue_insert_or_update
#define queue_clear local_array_queue_clear
#elif defined(_LOCAL_SPLAY_QUEUE)
#define queue_insert local_splay_queue_insert
#define queue_peek local_splay_queue_peek
#define queue_set_done local_splay_queue_set_done
#define queue_print local_splay_queue_print
#endif

//
// Queue interface:
//

// inserts a new event for node `ev.node` which occurs at timestamp `ev.ts` into the queue
__device__ bool queue_insert(queue_item item);

// Returns the index of the smallest element in the queue and sets the ev pointer to that event
#ifndef _HEAP_QUEUE
__device__ int queue_peek(queue_item **item, int lp);
#endif

// mark the event `index` as processed by the model. It will be removed on next queue cleanup.
__device__ void queue_set_done(int index);

// print the current queue content of logical process `lp` to standard out
__global__ void queue_print(int lp);

// returns whether there are any items enqueued for `lp`.
__device__ bool queue_is_empty(int lp);

// get the current number of queue_items enqueued for `lp`.
__device__ int queue_length(int lp);

// drop all elements currently enqueued for `lp`.
__device__ void queue_clear(int lp);

// Search for an identical item (via ==) in the queue and update it (via =), otherwise uses queue_insert.
__device__ void queue_insert_or_update(queue_item item, int lp);


#ifdef _COMPILE_LOCAL_ARRAY_QUEUE

  #include "local_array_queue.cuh"

#endif
#ifdef _COMPILE_LOCAL_HEAP_QUEUE

  #include "local_heap_queue.cuh"

#endif
#ifdef _COMPILE_HEAP_QUEUE

  #include "heap_queue.cuh"

#endif
#ifdef _COMPILE_GLOBAL_ARRAY_QUEUE

  #include "global_array_queue.cuh"

#endif
#ifdef _COMPILE_LOCAL_SPLAY_QUEUE

  #include "local_splay_queue.cuh"

#endif

