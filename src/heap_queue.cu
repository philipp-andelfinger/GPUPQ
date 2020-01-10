#include "config.h"
#include "queue.h"

#ifdef _COMPILE_HEAP_QUEUE

#include <stdio.h>
#include <moderngpu/kernel_mergesort.hxx>
#include "mgpu_caching_allocator_context.hxx"
#include <pthread.h>


#include "model.h"
#include "util.cuh"

#include "heap_queue.cuh"

struct HeapQueueData {
  bool merged_mark[HQ_NODE_SIZE + ENQUEUE_BUFFER_SIZE];
  bool safe_event_found[NUM_LPS];
  queue_item dequeue_buffer[HQ_NODE_SIZE];
  int peek_index;

  int event_sum;

  queue_item enqueue_buffer[ENQUEUE_BUFFER_SIZE + HQ_NODE_SIZE];
  // queue_item *enqueue_buffer;

  #ifdef HQ_DYNAMIC_ALLOC_FEL
  queue_item *fel;
  #else
  queue_item fel[HQ_NODE_SIZE * HQ_HEAP_SIZE];
  #endif
  int dequeue_count;

  int enqueue_count;
  int item_count[HQ_HEAP_SIZE];
  int last_node;
  int merge_buffer_size;
  int new_root_size;
  int num_delete_update;
  int num_insert_update;

  // delete-update
  bool mark_delete_update[HQ_HEAP_SIZE];


  #ifdef HQ_DYNAMIC_ALLOC_DELETE_UPDATE_BUFFER
  queue_item *delete_update_buffer[HQ_HEAP_SIZE];
  #else
  queue_item delete_update_buffer[HQ_HEAP_SIZE][3 * HQ_NODE_SIZE];
  #endif


  int delete_update_buffer_size[HQ_HEAP_SIZE];

  // insert-update
  #ifdef HQ_DYNAMIC_ALLOC_INSERT_PROCESS
  insert_info *insert_process;
  #else
  insert_info insert_process[NUM_INSERT_PROCESSES];
  #endif
  int current_insert_node;
  int current_insert_process;
  node_info insert_table[HQ_HEAP_SIZE];

  #ifdef HQ_DYNAMIC_ALLOC_INSERT_MERGE_BUFFER
  queue_item *insert_merge_buffer;
  #else
  queue_item insert_merge_buffer[NUM_INSERT_PROCESSES * 2 * HQ_NODE_SIZE];
  #endif

  int insert_count;
  int num_insert_curr;
  int num_insert_next;

  int copy_last_count;
  int copy_last_start;

  int min_idx[NUM_LPS];

  int num_threads_insert;
  int num_blocks_insert;
  int num_threads_node_d;
  int num_blocks_node_d;

  int num_new_processes;
  int sort_size[NUM_INSERT_PROCESSES];
};

__device__ HeapQueueData heap_queue_data[HQ_NUM_QUEUES];
HeapQueueData *heap_queue_data_h[HQ_NUM_QUEUES];

static int num_threads_fel = min(MAX_THREADS__HEAP_QUEUE, HQ_HEAP_SIZE);
static int num_blocks_fel = (HQ_HEAP_SIZE + num_threads_fel - 1) / num_threads_fel;
static int num_threads_node = min(MAX_THREADS__HEAP_QUEUE, HQ_NODE_SIZE);
static int num_blocks_node = (HQ_NODE_SIZE + num_threads_node - 1) / num_threads_node;
static int num_threads_ins_up = min(MAX_THREADS__HEAP_QUEUE, NUM_INSERT_PROCESSES);
static int num_blocks_ins_up = (NUM_INSERT_PROCESSES + num_threads_ins_up - 1) / num_threads_ins_up;


struct HostPointers {
  queue_item *host_insert_merge_buffer_ptr;
  queue_item *host_delete_update_buffer_ptr;
  queue_item *host_delete_update_buffer_array[HQ_HEAP_SIZE];
  bool *host_merged_mark_ptr;
  queue_item *host_enqueue_buffer_ptr;
  queue_item *host_fel_ptr;

  int *host_delete_update_buffer_size_ptr;
  bool *host_mark_delete_update_ptr;

  uint64_t *host_sort_size_ptr;
  int *int_ptr;
  queue_item *queue_item_ptr;
};

static HostPointers host_pointers[HQ_NUM_QUEUES];
// which queues to skip the next queue_post for
static bool *queue_post_mask;
static cudaStream_t stream[HQ_NUM_QUEUES];

#ifdef HQ_CACHING_ALLOCATOR
static caching_allocator_context_t *context[HQ_NUM_QUEUES];
#else
static mgpu::standard_context_t *context[HQ_NUM_QUEUES];
#endif

static cudaStream_t node_stream[HQ_NUM_QUEUES][HQ_HEAP_SIZE];
#ifdef HQ_CACHING_ALLOCATOR
static caching_allocator_context_t *node_context[HQ_NUM_QUEUES][HQ_HEAP_SIZE];
#else
static mgpu::standard_context_t *node_context[HQ_NUM_QUEUES][HQ_HEAP_SIZE];
#endif


__global__ void get_enqueue_count(int qid, int *ptr)
{
  *ptr = heap_queue_data[qid].enqueue_count;
}

__global__ void get_event_count_d(int qid, int node, int *ptr)
{
  *ptr = heap_queue_data[qid].item_count[node];
}


__global__ void get_merge_buffer_size(int qid, int *ptr)
{
  *ptr = heap_queue_data[qid].merge_buffer_size;
}

__global__ void get_dequeue_count(int qid, int *ptr)
{
  *ptr = heap_queue_data[qid].dequeue_count;
}

__global__ void get_dequeue_buffer_zero(int qid, queue_item *ptr)
{
  *ptr = heap_queue_data[qid].dequeue_buffer[0];
}

__global__ void get_item_count_zero(int qid, int *ptr)
{
  *ptr = heap_queue_data[qid].item_count[0];
}

__global__ void get_event_sum(int qid, int *ptr)
{
  *ptr = heap_queue_data[qid].event_sum;
}

__global__ void get_sort_size_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].sort_size;
}

__global__ void insert_update_post_fix_next_size(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];

  for(int idx = 0; idx < NUM_INSERT_PROCESSES; idx++)
  {
	  int node = q->insert_process[idx].current_node;
    if(q->insert_process[idx].next_size > 0)
    {
      int next_process = (idx + 1) % NUM_INSERT_PROCESSES;

      if(q->insert_process[next_process].current_node != node)
      {
        // printf("resetting for %d, %d\n", idx, (idx + 1) % NUM_INSERT_PROCESSES);
        q->insert_process[(idx + 1) % NUM_INSERT_PROCESSES].next_size = 0;
        q->insert_process[idx].size -= q->insert_process[idx].next_size;
        q->insert_process[idx].next_size = 0;
      }
    }
  }
  if(q->item_count[q->last_node + 1] != 0)
  {
    q->last_node++;
  }

}



__global__ void check_insert_processes(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  for(int i = 0; i < NUM_INSERT_PROCESSES; i++)
  {
    if (q->insert_process[i].current_node != -1) {
      if(q->insert_process[i].size < 0)
        printf("BUG! process %d has size %d, current_node is %d, target_node is %d\n", i, q->insert_process[i].size, q->insert_process[i].current_node, q->insert_process[i].target_node);
    }
  }
}

// -----------------------------
// Helper functions
// -----------------------------

int heap_queue_get_dequeue_count(int qid)
{
  int c = -1;

  get_dequeue_count<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&c, host_pointers[qid].int_ptr, sizeof(c), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  return c;
}

int heap_queue_get_enqueue_count(int qid)
{
  int c = -1;
  get_enqueue_count<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&c, host_pointers[qid].int_ptr, sizeof(c), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  return c;
}

static inline __host__ __device__ int get_level(int node)
{
  return floor(log2f(node + 1));
}

static inline __device__ int get_left_child_index(int node)
{
  return node * 2 + 1;
}

static inline __device__ const queue_item& get_max(int qid, int node)
{
  HeapQueueData *q = &heap_queue_data[qid];
  assert(q->item_count[node] > 0);
  return q->fel[node * HQ_NODE_SIZE + q->item_count[node] - 1];
}

static inline __device__ const queue_item& get_min(int qid, int node)
{
  HeapQueueData *q = &heap_queue_data[qid];
  assert(q->item_count[node] > 0);
  return q->fel[node * HQ_NODE_SIZE];
}

#ifdef _PHOLD
__device__ int search(int qid, queue_item *item, int count, long ts, bool flag_a)
{
  for (int i = 0; i < count; ++i)
  {
    if (flag_a && item[i].ts > ts) {
      return i;
    } else if (!flag_a && item[i].ts >= ts) {
      return i;
    }
  }

  return count;
}
#endif

static inline __device__ int get_next_insert_process(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  int next_process = ((q->current_insert_process + 1) % NUM_INSERT_PROCESSES);
  return next_process;
}

static int get_event_count(int qid, int node)
{
  int h_event_count;

  get_event_count_d<<<1, 1, 0, stream[qid]>>>(qid, node, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&h_event_count, host_pointers[qid].int_ptr, sizeof(h_event_count), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  return h_event_count;
}


__global__ void copy_enqueue_buffer(int qid, int process, int count)
{
  HeapQueueData *q = &heap_queue_data[qid];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx < count)
    q->enqueue_buffer[idx] = q->insert_process[process].insert_buffer[idx];
}

// -----------------------------
// Helper kernels
// -----------------------------

static __global__ void copy_buffer_to_node(int qid, int buf, int offset, int dest)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->item_count[dest];
			idx += blockDim.x * gridDim.x) {

		q->fel[dest * HQ_NODE_SIZE + idx] = q->delete_update_buffer[buf][offset + idx];

	}
}

static __global__ void copy_node_to_buffer(int qid, int node, int buffer_idx, int start_idx)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->item_count[node];
			idx += blockDim.x * gridDim.x) {

		q->delete_update_buffer[buffer_idx][start_idx + idx] = q->fel[node * HQ_NODE_SIZE + idx];
	}
}

static __global__ void copy_insert_buffer(int qid, int proc)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int node = q->insert_process[proc].current_node;

	if (idx < HQ_NODE_SIZE) {

		q->fel[node * HQ_NODE_SIZE + idx] = q->insert_merge_buffer[proc * 2 * HQ_NODE_SIZE + idx];

	} else if (idx < HQ_NODE_SIZE + q->insert_process[proc].size - q->insert_process[proc].next_size) {

		if (VERBOSE_DEBUG) {
#ifdef _PHOLD
			printf("setting insert_buffer[%d] = %ld\n",
					idx - HQ_NODE_SIZE, q->insert_merge_buffer[proc * 2 * HQ_NODE_SIZE + idx].ts);
#endif
		}

		q->insert_process[proc].insert_buffer[idx - HQ_NODE_SIZE] =
			q->insert_merge_buffer[proc * 2 * HQ_NODE_SIZE + idx];
	} else if (idx < HQ_NODE_SIZE + q->insert_process[proc].size) {

    int next_proc = (proc + 1) % NUM_INSERT_PROCESSES;

	  q->insert_process[proc].insert_buffer[idx - HQ_NODE_SIZE] = q->insert_process[next_proc].insert_buffer[idx - (HQ_NODE_SIZE + q->insert_process[proc].size - q->insert_process[proc].next_size)] =
	    q->insert_merge_buffer[proc * 2 * HQ_NODE_SIZE + idx];
  }
}

__device__ int heap_queue_length(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  return q->dequeue_count;
}

__device__ void heap_queue_clear(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	/* if (!threadIdx.x && !blockIdx.x)
  { */
		q->num_threads_insert = min(MAX_THREADS__HEAP_QUEUE, 2 * HQ_NODE_SIZE);
		q->num_blocks_insert = (2 * HQ_NODE_SIZE + q->num_threads_insert - 1) / q->num_threads_insert;

		q->num_threads_node_d = min(MAX_THREADS__HEAP_QUEUE, HQ_NODE_SIZE);
		q->num_blocks_node_d = (HQ_NODE_SIZE + q->num_threads_node_d - 1) / q->num_threads_node_d;

		q->current_insert_process = 0;
		q->current_insert_node = 0;
		q->num_insert_update = 0;
		q->num_delete_update = 0;

    q->dequeue_count = 0;
	// }

	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < HQ_HEAP_SIZE;
			idx += blockDim.x * gridDim.x)
  {
    q->item_count[idx] = 0;

    if(idx < NUM_INSERT_PROCESSES)
    {
		  q->insert_table[idx].offset = 0;
		  q->insert_process[idx].current_node = -1;
		  q->insert_process[idx].next_size = 0;
    }
	}

}


static __global__ void d_queue_init(int qid)
{
  queue_clear(qid);
}


// copy the first r keys to the dequeue buffer
static __global__ void update_dequeue(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->dequeue_count;
			idx += blockDim.x * gridDim.x) {

		if (VERBOSE_DEBUG) {
#ifdef _PHOLD
			printf("   [update_dequeue] setting q->dequeue_buffer[%d] = %ld\n", idx,
					q->enqueue_buffer[idx].ts);
#endif
		}

		q->dequeue_buffer[idx] = q->enqueue_buffer[idx];
	}
}

// copy the second r keys to the root node
static __global__ void update_root(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// mark root node for delete-update process
	if (idx == 0) {
    // printf("setting root size to %d\n", q->new_root_size);
		q->item_count[0] = q->new_root_size;
		q->insert_table[0].offset = q->new_root_size;
		q->mark_delete_update[0] = true;
		atomicAdd(&q->num_delete_update, 1);

		if (q->current_insert_node == 0 && q->item_count[0] == HQ_NODE_SIZE) {
			q->current_insert_node++;
		}
	}

	for (; idx < q->new_root_size; idx += blockDim.x * gridDim.x) {

		if (VERBOSE_DEBUG) {
#ifdef _PHOLD
			printf("update_root: setting q->fel[%d] = q->enqueue_buffer[%d] (= %ld)\n", idx,
					q->dequeue_count + idx, q->enqueue_buffer[q->dequeue_count + idx].ts);
#endif
		}

		q->fel[idx] = q->enqueue_buffer[q->dequeue_count + idx];
	}
}


static __global__ void d_queue_check_heap(int qid)
{
}

static void heap_queue_check_heap(int qid)
{
	d_queue_check_heap<<<1, 1, 0, stream[qid]>>>(qid);
}

__global__ void partition_merged_parallel_init(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < NUM_LPS) {
		q->safe_event_found[i] = false;
		q->min_idx[i] = INT_MAX;
	}

	if (i < q->merge_buffer_size) {
		q->merged_mark[i] = false;
	}

	if(!blockIdx.x && !threadIdx.x)
		q->dequeue_count = 0;
}

__global__ void partition_merged_parallel_min(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i >= q->merge_buffer_size)
		return;

#ifdef _PHOLD
	long d_current_ts = q->enqueue_buffer[0].ts;
	if (q->enqueue_buffer[i].ts >= d_current_ts + LOOKAHEAD) {
		return;
	}
#endif

	/* prevent dequeueing the wrong event if the next event for an LP is
	 * not in the root but in a child node */
	if (q->item_count[0] > 0 && q->enqueue_buffer[i] > get_max(qid, 0)) {
		return;
	}

	int lp = get_lp(q->enqueue_buffer[i].node);

	int old_min_idx = atomicMin(&q->min_idx[lp], i);
	q->safe_event_found[lp] = true;

}

__global__ void partition_merged_parallel_mark(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int lp = threadIdx.x + blockDim.x * blockIdx.x;

	if(lp > NUM_LPS)
		return;

	if(q->min_idx[lp] != INT_MAX)
	{
		q->merged_mark[q->min_idx[lp]] = true;
		atomicAdd(&q->dequeue_count, 1);

	}
}

__global__ void partition_merged_parallel_update_counts(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	q->new_root_size = min(HQ_NODE_SIZE, q->merge_buffer_size - q->dequeue_count);
	q->enqueue_count = q->merge_buffer_size - q->dequeue_count - q->new_root_size;


}

static void process_heap(int qid)
{
	// delete-update at even nodes (incl. root)
  CudaCheckError();
	delete_update_pre<<<num_blocks_fel, num_threads_fel, 0, stream[qid]>>>(qid, true);
  cudaStreamSynchronize(stream[qid]);
	delete_update(qid, true);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();


#ifdef HQ_CACHING_ALLOCATOR
  cudaStreamSynchronize(stream[qid]);
#endif

	delete_update_post<<<num_blocks_fel, num_threads_fel, 0, stream[qid]>>>(qid, true);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();


	// insert-update at even nodes (incl. root)
	insert_update_pre<<<num_blocks_ins_up, num_threads_ins_up, 0, stream[qid]>>>(qid, true);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();
	insert_update(qid);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();
	insert_update_post<<<num_blocks_ins_up, num_threads_ins_up, 0, stream[qid]>>>(qid, true);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();

	insert_update_post_fix_next_size<<<1, 1, 0, stream[qid]>>>(qid);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();


	// delete-update at odd nodes
	delete_update_pre<<<num_blocks_fel, num_threads_fel, 0, stream[qid]>>>(qid, false);
  cudaStreamSynchronize(stream[qid]);
  CudaCheckError();
	delete_update(qid, false);

#ifdef HQ_CACHING_ALLOCATOR
  cudaStreamSynchronize(stream[qid]);
#endif

	delete_update_post<<<num_blocks_fel, num_threads_fel, 0, stream[qid]>>>(qid, false);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	// insert-update at odd nodes
	insert_update_pre<<<num_blocks_ins_up, num_threads_ins_up, 0, stream[qid]>>>(qid, false);
  cudaStreamSynchronize(stream[qid]);
	insert_update(qid);
  cudaStreamSynchronize(stream[qid]);
  CudaCheckError();
	insert_update_post<<<num_blocks_ins_up, num_threads_ins_up, 0, stream[qid]>>>(qid, false);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();

	insert_update_post_fix_next_size<<<1, 1, 0, stream[qid]>>>(qid);
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();


#ifdef _CORRECTNESS_CHECKS
	heap_queue_check_heap();
#endif

}



// -----------------------------
// Main kernels
// -----------------------------

__global__ void delete_insert()
{

}

__global__ void delete_update_pre(int qid, bool even)
{
  HeapQueueData *q = &heap_queue_data[qid];
  q->peek_index = 0;
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < HQ_HEAP_SIZE;
			idx += blockDim.x * gridDim.x) {

		int level = get_level(idx);
		bool level_even = !(level % 2);

		if (q->mark_delete_update[idx] && even == level_even) {

			int left_child = get_left_child_index(idx);
			int right_child = left_child + 1;

			/* don't continue the delete-update process */
			/* if bottom of heap is reached ... */
			if ((left_child >= HQ_HEAP_SIZE || q->item_count[left_child] == 0)
					&& (right_child >= HQ_HEAP_SIZE || q->item_count[right_child] == 0)) {
				q->mark_delete_update[idx] = false;
				atomicSub(&q->num_delete_update, 1);
				continue;
			}

			if ((left_child >= HQ_HEAP_SIZE || q->item_count[left_child] == 0 || get_max(qid, idx) <= get_min(qid, left_child))
					&& (right_child >= HQ_HEAP_SIZE || q->item_count[right_child] == 0 || get_max(qid, idx) <= get_min(qid, right_child)))
			{
				q->mark_delete_update[idx] = false;
				atomicSub(&q->num_delete_update, 1);
				continue;
			}

			copy_node_to_buffer<<<q->num_blocks_node_d, q->num_threads_node_d>>>
				(qid, idx, idx, 0);
			copy_node_to_buffer<<<q->num_blocks_node_d, q->num_threads_node_d>>>
				(qid, left_child, idx, q->item_count[idx]);
			copy_node_to_buffer<<<q->num_blocks_node_d, q->num_threads_node_d>>>
				(qid, right_child, idx, q->item_count[idx] + q->item_count[left_child]);


			q->delete_update_buffer_size[idx] = q->item_count[idx] +
				q->item_count[left_child] + q->item_count[right_child];
		}
	}
}

void delete_update(int qid, bool even)
{
	bool h_mark_delete_update[HQ_HEAP_SIZE];
	int h_delete_update_buffer_size[HQ_HEAP_SIZE];

	CudaSafeCall( cudaMemcpyAsync(h_mark_delete_update, host_pointers[qid].host_mark_delete_update_ptr,
				sizeof(h_mark_delete_update), cudaMemcpyDeviceToHost, stream[qid]) );

	CudaSafeCall( cudaMemcpyAsync(h_delete_update_buffer_size,
				host_pointers[qid].host_delete_update_buffer_size_ptr, sizeof(h_delete_update_buffer_size), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  static int stream_active[HQ_HEAP_SIZE];
	for (int i = 0; i < HQ_HEAP_SIZE; ++i) {

		int level = get_level(i);
		bool level_even = !(level % 2);

		if (h_mark_delete_update[i] && even == level_even) {

			queue_item *host_ptr;
#ifdef HQ_DYNAMIC_ALLOC_DELETE_UPDATE_BUFFER
			host_ptr = host_pointers[qid].host_delete_update_buffer_array[i];
#else
			cudaGetSymbolAddress((void **)&host_ptr, q->delete_update_buffer);
			host_ptr += i * 3 * HQ_NODE_SIZE; // q->delete_update_buffer is actually two-dimensional...
#endif

      stream_active[i] = true;

#ifdef HQ_MGPU_STREAM_PER_NODE
			mgpu::mergesort(host_ptr, h_delete_update_buffer_size[i], mgpu::less_t<queue_item>(), *(node_context[qid][i]));
#else
			mgpu::mergesort(host_ptr, h_delete_update_buffer_size[i], mgpu::less_t<queue_item>(), *(context[qid]));
#endif

		}
	}

#ifdef HQ_MGPU_STREAM_PER_NODE
  for(int i = 0; i < HQ_HEAP_SIZE; i++) {
    if(stream_active[i])
    {
      cudaStreamSynchronize(node_stream[qid][i]);
      stream_active[i] = false;
    }
  }
#endif
}

__global__ void delete_update_post(int qid, bool even)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < HQ_HEAP_SIZE;
			idx += blockDim.x * gridDim.x) {

		int level = get_level(idx);
		bool level_even = !(level % 2);

		if (q->mark_delete_update[idx] && even == level_even) {

			q->mark_delete_update[idx] = false;

			int left_child = get_left_child_index(idx);
			int right_child = left_child + 1;

			copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
				(qid, idx, 0, idx);

			if (right_child >= HQ_HEAP_SIZE || q->item_count[right_child] == 0) {

				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE, left_child);

				q->mark_delete_update[left_child] = true;

			} else if (left_child >= HQ_HEAP_SIZE || q->item_count[left_child] == 0) {

				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE, right_child);

				q->mark_delete_update[right_child] = true;

			} else if (q->item_count[right_child] < HQ_NODE_SIZE
					|| get_max(qid, right_child) > get_max(qid, left_child)) {

				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE, right_child);
				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE + q->item_count[right_child], left_child);

				q->mark_delete_update[left_child] = true;

			} else {

				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE, left_child);
				copy_buffer_to_node<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx, HQ_NODE_SIZE + q->item_count[left_child], right_child);

				q->mark_delete_update[right_child] = true;

			}
		}
	}
}

__global__ void copy_node_to_insert_buffer(int qid, int node, int process)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int size = q->insert_process[process].size;

	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->item_count[node];
			idx += blockDim.x * gridDim.x) {

		q->insert_merge_buffer[process * 2 * HQ_NODE_SIZE + size + idx] =
			q->fel[node * HQ_NODE_SIZE + idx];

	}
}

__global__ void copy_process_to_insert_buffer(int qid, int process)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->insert_process[process].size;
			idx += blockDim.x * gridDim.x) {

		q->insert_merge_buffer[process * 2 * HQ_NODE_SIZE + idx] =
			q->insert_process[process].insert_buffer[idx];

	}
}




__global__ void insert_update_pre(int qid, bool even)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < NUM_INSERT_PROCESSES;
			idx += blockDim.x * gridDim.x) {

		int node = q->insert_process[idx].current_node;

		if (node != -1 && q->insert_process[idx].next_size != -1) {

			int level = get_level(node);
			bool level_even = !(level % 2);

			if (even == level_even) {

				copy_node_to_insert_buffer<<<q->num_blocks_node_d, q->num_threads_node_d>>>
				  (qid, node, idx);
				copy_process_to_insert_buffer<<<q->num_blocks_node_d, q->num_threads_node_d>>>
					(qid, idx);

				q->sort_size[idx] = q->item_count[node] + q->insert_process[idx].size;

			}
		}
	}
}

__global__ void dump_mb(int qid, int buf)
{
  /* HeapQueueData *q = &heap_queue_data[qid];
  for(int i = 0; i < q->sort_size[buf]; i++)
  {
    queue_item item = q->insert_merge_buffer[buf * 2 * HQ_NODE_SIZE + i];
    printf("%d: %d: %d, %ld\n", buf, i, item.node, item.ts);
  } */
}

/* if q->sort_size is > 0, the insert_merged_buffer will be sorted (will only be
 * set for even or odd levels, respectively) */
void insert_update(int qid)
{
	int h_sort_size[NUM_INSERT_PROCESSES];

  CudaCheckError();
	CudaSafeCall( cudaMemcpyAsync(h_sort_size, host_pointers[qid].host_sort_size_ptr, sizeof(int) * NUM_INSERT_PROCESSES, cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);
  CudaCheckError();

#ifdef HQ_DYNAMIC_ALLOC_INSERT_MERGE_BUFFER
	queue_item *host_ptr = host_pointers[qid].host_insert_merge_buffer_ptr;
#else
	queue_item *host_ptr;
  cudaGetSymbolAddress((void **)&host_ptr, q->insert_merge_buffer);
#endif
  CudaCheckError();

  static bool stream_active[HQ_NUM_QUEUES][HQ_HEAP_SIZE];
	for (int i = 0; i < NUM_INSERT_PROCESSES; ++i) {
		if (h_sort_size[i] > 0) {

			queue_item *ptr = host_ptr + i * 2 * HQ_NODE_SIZE;

#ifdef HQ_MGPU_STREAM_PER_NODE
      stream_active[qid][i] = true;
			mgpu::mergesort(ptr, h_sort_size[i], mgpu::less_t<queue_item>(), *(node_context[qid][i]));
#else
			mgpu::mergesort(ptr, h_sort_size[i], mgpu::less_t<queue_item>(), *(context[qid]));
#endif
      CudaCheckError();

		}
	}

#ifdef HQ_MGPU_STREAM_PER_NODE
  for(int i = 0; i < HQ_HEAP_SIZE; i++) {
    if(stream_active[qid][i])
    {
      cudaStreamSynchronize(node_stream[qid][i]);
      stream_active[qid][i] = false;
      CudaCheckError();
    }
  }
#endif
}

__global__ void insert_update_post(int qid, bool even)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < NUM_INSERT_PROCESSES;
			idx += blockDim.x * gridDim.x) {

		int node = q->insert_process[idx].current_node;

		if (node != -1) {

			int level = get_level(node);
			bool level_even = !(level % 2);

			if (even == level_even) {

				q->sort_size[idx] = 0;


        if(q->insert_process[idx].next_size != -1)
        {
				  copy_insert_buffer<<<q->num_blocks_insert, q->num_threads_insert>>>(qid, idx);

        } else {
        }
			  CudaSafeCall( cudaDeviceSynchronize() ); // this is necessary!
			  CudaCheckError();

				int level_diff = get_level(q->insert_process[idx].target_node) - get_level(node);
				bool move_left = !((q->insert_process[idx].target_node+1) & (1 << (level_diff-1)));

				if (VERBOSE_DEBUG) {
					/* printf("   [insert-update for process %d] current node: %d, target node: %d, move_left: %d, first element: %ld\n",
							idx, node, q->insert_process[idx].target_node, move_left, q->insert_process[idx].insert_buffer[0].ts); */
				}

				if (q->insert_process[idx].current_node == q->insert_process[idx].target_node) {
						q->last_node = q->insert_process[idx].current_node;
						q->insert_process[idx].current_node = -1;
            if(q->insert_process[idx].next_size > 0)
              q->insert_process[idx].size -= q->insert_process[idx].next_size;
						q->item_count[node] += q->insert_process[idx].size;
						q->insert_process[idx].next_size = 0;
						atomicSub(&q->num_insert_update, 1);

				} else {

					q->insert_process[idx].current_node = move_left ?
						get_left_child_index(node) : get_left_child_index(node) + 1;


          if(q->insert_process[idx].current_node > HQ_HEAP_SIZE)
            printf("BUG! process moving beyond heap bounds\n");
				}
			}
		}
	}
}

/* only used inside the copy_last kernels */
__global__ void copy_insert_buffer_to_root(int qid, int insert_process_index, int src_offset, int dest_offset, int count)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < count;
			idx += blockDim.x * gridDim.x) {

		q->fel[dest_offset + idx] = q->insert_process[insert_process_index].insert_buffer[src_offset + idx];
	}

}

static __global__ void copy_last_pre(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int required = HQ_NODE_SIZE - q->item_count[0];

	if (required == 0 || q->last_node == 0) {
		q->copy_last_count = 0;
		return;
	}

  // this one's a bit tricky: collect 'required' items from those insert buffers targeting the end of the heap, and set new q->current_insert_node
  int  current_process = (get_next_insert_process(qid) + NUM_INSERT_PROCESSES - 1) % NUM_INSERT_PROCESSES;
  for(int i = 0; i < NUM_INSERT_PROCESSES && required > 0; i++)
  {
    if(q->insert_process[current_process].current_node != -1)
    {
      int remove_count = min(required, q->insert_process[current_process].size);

      if(q->insert_process[current_process].next_size == -1)
      {
        // special case: need to update previous process as well
        int prev_process = (current_process + NUM_INSERT_PROCESSES - 1) % NUM_INSERT_PROCESSES;
        q->insert_process[prev_process].size -= remove_count;
        q->insert_process[prev_process].next_size -= remove_count;
      }

      int num_blocks = ceil((float)remove_count / MAX_THREADS__HEAP_QUEUE);
      copy_insert_buffer_to_root<<<num_blocks, MAX_THREADS__HEAP_QUEUE>>>(qid, current_process, q->insert_process[current_process].size - remove_count, q->item_count[0], remove_count);

      q->insert_process[current_process].size -= remove_count;
      q->insert_table[q->insert_process[current_process].target_node].offset -= remove_count;

      if(q->insert_process[current_process].size == 0)
      {
        q->insert_process[current_process].current_node = -1;
        q->insert_process[current_process].next_size = 0;
      }

      q->current_insert_node = q->insert_process[current_process].target_node;

      q->item_count[0] += remove_count;
      q->insert_table[0].offset += remove_count;
      required -= remove_count;
    }
    current_process = (current_process + NUM_INSERT_PROCESSES - 1) % NUM_INSERT_PROCESSES;
  }

  // the remaining items come from the last and next to last node
	if (required == 0 || q->last_node == 0) {
		q->copy_last_count = 0;
		return;
	}

	int count1 = 0;
	int count2 = 0;

	if (q->last_node > 0) {
		count1 = min(required, q->item_count[q->last_node]);
		required -= count1;
	}

	if (required > 0 && q->last_node - 1 > 0) {
		count2 = min(required, q->item_count[q->last_node - 1]);
		required -= count2;
	}

	q->copy_last_count = count1 + count2;
	q->copy_last_start = q->last_node * HQ_NODE_SIZE + q->item_count[q->last_node] -
		q->copy_last_count;

	q->item_count[0] += count1;
	q->item_count[q->last_node] -= count1;
	q->insert_table[0].offset += count1;
	q->insert_table[q->last_node].offset -= count1;

	q->item_count[0] += count2;
	q->item_count[q->last_node - 1] -= count2;
	q->insert_table[0].offset += count2;
	q->insert_table[q->last_node - 1].offset -= count2;

	if (q->insert_table[q->last_node].offset < HQ_NODE_SIZE) {
		q->current_insert_node = q->last_node;
	}

	if (q->insert_table[q->last_node - 1].offset < HQ_NODE_SIZE) {
		q->current_insert_node = q->last_node - 1;
	}

	if (q->item_count[q->last_node] == 0) {
		q->last_node--;
	}
}

static __global__ void copy_last_copy(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->copy_last_count;
			idx += blockDim.x * gridDim.x) {

		q->fel[HQ_NODE_SIZE - q->copy_last_count + idx] = q->fel[q->copy_last_start + idx];
	}
}

void copy_last(int qid)
{
	copy_last_pre<<<1, 1, 0, stream[qid]>>>(qid);
	CudaCheckError();

  cudaStreamSynchronize(stream[qid]);

	copy_last_copy<<<num_blocks_node, num_threads_node, 0, stream[qid]>>>(qid);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	/* sort the root node */
  if(get_event_count(qid, 0) > 0)
  	mgpu::mergesort(host_pointers[qid].host_fel_ptr, get_event_count(qid, 0), mgpu::less_t<queue_item>(), *(context[qid]));
  CudaCheckError();
}

/* only used inside the init_insert_update kernels */
static __global__ void init_insert_update_pre(int qid, int insert_source, int source_offset)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx == 0) {
		q->insert_count = min(q->enqueue_count, HQ_NODE_SIZE);

		// currently inserts up to HQ_NODE_SIZE items per enqueue operation

		int free_space = HQ_NODE_SIZE - q->insert_table[q->current_insert_node].offset;

		q->num_insert_curr = min(q->insert_count, free_space);
		q->num_insert_next = q->insert_count - q->num_insert_curr;

    q->num_new_processes = 0;
		if (q->num_insert_curr > 0) {
			q->insert_table[q->current_insert_node].offset += q->num_insert_curr;
			q->insert_process[q->current_insert_process].current_node = 0;
			q->insert_process[q->current_insert_process].target_node = q->current_insert_node;
			q->insert_process[q->current_insert_process].size = q->num_insert_curr + q->num_insert_next;
      q->insert_process[q->current_insert_process].next_size = q->num_insert_next;
			atomicAdd(&q->num_insert_update, 1);

      q->num_new_processes++;
		}

		if (q->insert_table[q->current_insert_node].offset == HQ_NODE_SIZE) {
			q->current_insert_node++;
		}

		if (q->num_insert_next > 0) {
			q->insert_table[q->current_insert_node].offset += q->num_insert_next;

      int next_insert_process = get_next_insert_process(qid);
      if(q->insert_process[next_insert_process].current_node != -1)
      {
        printf("BUG! new insert process non-empty\n");
      }

			q->insert_process[next_insert_process].current_node = 0;
			q->insert_process[next_insert_process].target_node = q->current_insert_node;
      q->insert_process[next_insert_process].next_size = -1;
			q->insert_process[next_insert_process].size = q->num_insert_next;

      q->num_insert_curr += q->num_insert_next;
      q->num_insert_next = 0;

      q->num_new_processes++;

			atomicAdd(&q->num_insert_update, 1);
		}

		if (q->insert_table[q->current_insert_node].offset == HQ_NODE_SIZE) {
			q->current_insert_node++;
		}

		if (q->current_insert_node >= HQ_HEAP_SIZE) {
			printf("Error: Queue is full.\n");
		}
	}
}

static __global__ void init_insert_update_copy(int qid, int insert_source, int source_offset)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->insert_count;
			idx += blockDim.x * gridDim.x) {

		queue_item *ptr;
		int offset;

		if (insert_source == INSERT_FROM_ENQUEUE_BUFFER) {
			ptr = q->enqueue_buffer;
			offset = 0;
		} else if (insert_source == INSERT_FROM_MERGED_BUFFER) {
			ptr = q->enqueue_buffer;
			offset = q->dequeue_count + HQ_NODE_SIZE;
		}

		if (idx < q->num_insert_curr) {

			if (VERBOSE_DEBUG) {
#ifdef _PHOLD
				printf("init_insert_update: copy idx %d (= %ld) --> insert buffer %d at pos %d\n",
						source_offset + offset + idx, ptr[source_offset + offset + idx].ts,
						q->current_insert_process, idx);
#endif
			}

			q->insert_process[q->current_insert_process].insert_buffer[idx] =
				ptr[source_offset + offset + idx];

		} else if (idx - q->num_insert_curr < q->num_insert_next) {

			if (VERBOSE_DEBUG) {
#ifdef _PHOLD
				printf("init_insert_update: copy idx %d (= %ld) --> insert buffer %d at pos %d\n",
						source_offset + offset + idx, ptr[source_offset + offset + idx].ts,
						get_next_insert_process(qid), idx - q->num_insert_curr);
#endif
			}

			q->insert_process[get_next_insert_process(qid)].insert_buffer[idx -
				q->num_insert_curr] = ptr[source_offset + offset + idx];

		} else {

			if (VERBOSE_DEBUG) {
				printf("init_insert_update: not copying at idx %d, q->num_insert_curr: %d, q->num_insert_next: %d\n",
						idx, q->num_insert_curr, q->num_insert_next);
			}
		}
	}
}

static __global__ void init_insert_update_post(int qid, int insert_source, int source_offset)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx == 0) {

    while(q->num_new_processes--)
      q->current_insert_process = get_next_insert_process(qid);

		q->enqueue_count -= q->insert_count;
	}

}

void init_insert_update(int qid, int insert_source, int source_offset /* = 0 */)
{
	/* insert_source == 0 for q->enqueue_buffer, 1 for merged buffer */

	CudaCheckError();
	init_insert_update_pre<<<1,1, 0, stream[qid]>>>
		(qid, insert_source, source_offset);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	init_insert_update_copy<<<num_blocks_node, num_threads_node, 0, stream[qid]>>>
		(qid, insert_source, source_offset);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	init_insert_update_post<<<1,1, 0, stream[qid]>>>
		(qid, insert_source, source_offset);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

}





// -----------------------------
// Queue interface
// -----------------------------

__global__ void set_enqueue_buffer_ptr(int qid, queue_item *ptr)
{
  // heap_queue_data[qid].enqueue_buffer = ptr;
}

__global__ void get_enqueue_buffer_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].enqueue_buffer;
}

__global__ void get_merged_mark_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].merged_mark;
}

__global__ void get_delete_update_buffer_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].delete_update_buffer;
}

__global__ void get_delete_update_buffer_size_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].delete_update_buffer_size;
}

__global__ void get_mark_delete_update_ptr(int qid, uint64_t *ptr)
{
  *ptr = (uint64_t)heap_queue_data[qid].mark_delete_update;
}
__global__ void set_fel_ptr(int qid, queue_item *ptr)
{
  heap_queue_data[qid].fel = ptr;
}

__global__ void set_insert_merge_buffer_ptr(int qid, queue_item *ptr)
{
  heap_queue_data[qid].insert_merge_buffer = ptr;
}

__global__ void set_insert_process_ptr(int qid, insert_info *ptr)
{
  heap_queue_data[qid].insert_process = ptr;
}



void heap_queue_init()
{
#ifdef _PHOLD
  assert(HQ_NODE_SIZE >= NUM_LPS); // we can only insert at most HQ_NODE_SIZE items per iteration
#endif

  cudaGetSymbolAddress((void **)(&heap_queue_data_h[0]), heap_queue_data[0]);


  for(int i = 1; i < HQ_NUM_QUEUES; i++)
  {
    heap_queue_data_h[i] = heap_queue_data_h[i - 1] + sizeof(HeapQueueData);
  }


  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {
#ifdef HQ_MGPU_STREAM_PER_NODE
    for(int i = 0; i < HQ_HEAP_SIZE; i++)
    {
      cudaStreamCreate(&node_stream[qid][i]);

    #ifdef HQ_CACHING_ALLOCATOR
      node_context[qid][i] = new caching_allocator_context_t(node_stream[qid][i]);
    #else
      node_context[qid][i] = new mgpu::standard_context_t(false, node_stream[qid][i]);
    #endif

      CudaCheckError();
    }
#endif

    cudaStreamCreate(&stream[qid]);

  #ifdef HQ_CACHING_ALLOCATOR
    context[qid] = new caching_allocator_context_t(stream[qid]);
  #else
    context[qid] = new mgpu::standard_context_t(false, stream[qid]);
  #endif
    CudaCheckError();

  // Set corresponding device_ptrs to __device__ variables

  cudaMalloc((void **)&host_pointers[qid].int_ptr, sizeof(int)); // for getters
  cudaMalloc((void **)&host_pointers[qid].queue_item_ptr, sizeof(queue_item)); // for getters

  uint64_t *ptr_d, ptr_h;
  cudaMalloc((void **)&ptr_d, sizeof(uint64_t));


  cudaMalloc((void **)&ptr_d, sizeof(uint64_t));
  CudaCheckError();
  get_enqueue_buffer_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&ptr_h, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();
  host_pointers[qid].host_enqueue_buffer_ptr = (queue_item *)ptr_h;

  get_merged_mark_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&ptr_h, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();
  host_pointers[qid].host_merged_mark_ptr = (bool *)ptr_h;

  get_sort_size_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&host_pointers[qid].host_sort_size_ptr, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();

  get_delete_update_buffer_size_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&host_pointers[qid].host_delete_update_buffer_size_ptr, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();

  get_mark_delete_update_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&host_pointers[qid].host_mark_delete_update_ptr, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();


#ifdef HQ_DYNAMIC_ALLOC_FEL
  CudaSafeCall( cudaMalloc((void **)&host_pointers[qid].host_fel_ptr, sizeof(queue_item) * HQ_NODE_SIZE * HQ_HEAP_SIZE) );
  set_fel_ptr<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].host_fel_ptr);
#else
  cudaGetSymbolAddress((void **)&host_fel_ptr, q->fel);
#endif


#ifdef HQ_DYNAMIC_ALLOC_INSERT_MERGE_BUFFER
  CudaSafeCall( cudaMalloc((void **)&host_pointers[qid].host_insert_merge_buffer_ptr, sizeof(queue_item) * NUM_INSERT_PROCESSES * 2 * HQ_NODE_SIZE) );
  set_insert_merge_buffer_ptr<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].host_insert_merge_buffer_ptr);
#endif

#ifdef HQ_DYNAMIC_ALLOC_DELETE_UPDATE_BUFFER
  for(int i = 0; i < HQ_HEAP_SIZE; i++) {
    CudaSafeCall ( cudaMalloc(&host_pointers[qid].host_delete_update_buffer_array[i], 3 * HQ_NODE_SIZE * sizeof(queue_item)) );
  }

  get_delete_update_buffer_ptr<<<1, 1, 0, stream[qid]>>>(qid, ptr_d);
  cudaMemcpy((void *)&ptr_h, ptr_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CudaCheckError();
  host_pointers[qid].host_delete_update_buffer_ptr = (queue_item *)ptr_h;

  CudaSafeCall ( cudaMemcpy(host_pointers[qid].host_delete_update_buffer_ptr, host_pointers[qid].host_delete_update_buffer_array, HQ_HEAP_SIZE * sizeof(queue_item *), cudaMemcpyHostToDevice) );
  CudaCheckError();
#endif

  insert_info *host_insert_process_ptr;
#ifdef HQ_DYNAMIC_ALLOC_INSERT_PROCESS
  CudaSafeCall( cudaMalloc((void **)&host_insert_process_ptr, NUM_INSERT_PROCESSES * sizeof(insert_info)) );

  set_insert_process_ptr<<<1, 1, 0, stream[qid]>>>(qid, host_insert_process_ptr);

#endif


  d_queue_init<<<num_blocks_node, num_threads_node, 0, stream[qid]>>>(qid);
  CudaSafeCall( cudaDeviceSynchronize() );
  CudaCheckError();

  }
}

void heap_queue_finish()
{
}

__global__ void update_counts(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  q->dequeue_count = min(q->merge_buffer_size, HQ_NODE_SIZE);
  // printf("update_counts, new dequeue_count: %d, %d\n", q->dequeue_count, q->merge_buffer_size);
	q->new_root_size = min(HQ_NODE_SIZE, q->merge_buffer_size - q->dequeue_count);
	q->enqueue_count = q->merge_buffer_size - q->dequeue_count - q->new_root_size;
}

void set_queue_post_mask(bool *mask) {
  queue_post_mask = mask;
}


void sort_enqueue_buffer(int qid)
{
  if(heap_queue_get_enqueue_count(qid) > 0)
    mgpu::mergesort(host_pointers[qid].host_enqueue_buffer_ptr, heap_queue_get_enqueue_count(qid), mgpu::less_t<queue_item>(), *(context[qid]));

}

bool sort_merged_buffer(int qid)
{
  int size;

  get_merge_buffer_size<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&size, host_pointers[qid].int_ptr, sizeof(size), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  CudaCheckError();

  if(size == 0)
    return false;


  // printf("size is %d, ebp is %p\n", size, host_pointers[qid].host_enqueue_buffer_ptr);
  if(size == 1)
    return true;

  mgpu::mergesort(host_pointers[qid].host_enqueue_buffer_ptr, size, mgpu::less_t<queue_item>(), *(context[qid]));
  CudaCheckError();

  return true;
}

void sort_merged_buffer_marked(int qid)
{
  int h_merge_buffer_size;

  get_merge_buffer_size<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&h_merge_buffer_size, host_pointers[qid].int_ptr, sizeof(h_merge_buffer_size), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);

  mgpu::mergesort(host_pointers[qid].host_merged_mark_ptr, host_pointers[qid].host_enqueue_buffer_ptr, h_merge_buffer_size, mgpu::greater_t<bool>(),
      *(context[qid]));
}


__global__ void reset_enqueue_buffer(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
  queue_item zero_item = {-1, -1};
  q->enqueue_buffer[idx] = zero_item;
  if(!idx)
    q->enqueue_count = 0;
}


void heap_queue_post_init()
{
  cudaDeviceSynchronize();
  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {

  	sort_enqueue_buffer(qid);
    CudaCheckError();

  	init_insert_update(qid, INSERT_FROM_ENQUEUE_BUFFER, 0);
    CudaCheckError();

  	process_heap(qid);
    CudaCheckError();

    reset_enqueue_buffer<<<ceil((float)ENQUEUE_BUFFER_SIZE / 256), 256, 0, stream[qid]>>>(qid);
  }
  cudaDeviceSynchronize();
}
void heap_queue_pre()
{
  cudaDeviceSynchronize();
  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {

    reset_enqueue_buffer<<<ceil((float)ENQUEUE_BUFFER_SIZE / 256), 256, 0, stream[qid]>>>(qid);

  }
  cudaDeviceSynchronize();
}

static __global__ void copy_root_to_merged(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
			idx < q->item_count[0];
			idx += blockDim.x * gridDim.x) {

		q->enqueue_buffer[q->enqueue_count + idx] = q->fel[idx];

	}
}

// append the elements of the heap's root node to the q->enqueue_buffer,
// and store the size of the combined buffers in q->merge_buffer_size.
static __global__ void merge_fel_enqueue(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  q->enqueue_count = min(ENQUEUE_BUFFER_SIZE, q->enqueue_count);
	q->merge_buffer_size = q->enqueue_count + q->item_count[0];
  // printf("%d %d %d -> %d\n", ENQUEUE_BUFFER_SIZE, q->enqueue_count, q->item_count[0],  q->merge_buffer_size);

  if(q->enqueue_count > ENQUEUE_BUFFER_SIZE)
  {
    printf("q->enqueue_count is broken\n");
  }
	copy_root_to_merged<<<q->num_blocks_node_d, q->num_threads_node_d>>>(qid);
}

// Partitions the merged buffer into (dequeue, new_root, insert) based on
// current_ts by setting:
// q->dequeue_count: the range size of the elements dequeued from the heap for the next queue_peek() calls
// q->new_root_size: the range size of the elements which make up the new root
// q->enqueue_count: thet range size of the elements which belong into child nodes.
__global__ void partition_merged(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];

	// initialize q->safe_event_found and q->merged_mark with false
	for (int i = 0; i < max(NUM_LPS, q->merge_buffer_size); ++i) {
		if (i < NUM_LPS) {
			q->safe_event_found[i] = false;
		}

		if (i < q->merge_buffer_size) {
			q->merged_mark[i] = false;
		}
	}

	q->dequeue_count = 0;

  printf("in partition_merged(): q->merge_buffer_size: %d\n", q->merge_buffer_size);
	for (int i = 0; i < q->merge_buffer_size; ++i) {

#ifdef _PHOLD
		long d_current_ts = q->enqueue_buffer[0].ts;
		if (q->enqueue_buffer[i].ts >= d_current_ts + LOOKAHEAD) {
			break;
		}
#endif

		/* prevent dequeueing the wrong event if the next event for an LP is
		 * not in the root but in a child node */
		if (q->item_count[0] > 0 && q->enqueue_buffer[i] > get_max(qid, 0)) {
			break;
		}

		int lp = get_lp(q->enqueue_buffer[i].node);
		if (!q->safe_event_found[lp]) {
			q->dequeue_count++;
			q->merged_mark[i] = true;
			q->safe_event_found[lp] = true;
		}

	}

	q->new_root_size = min(HQ_NODE_SIZE, q->merge_buffer_size - q->dequeue_count);
	q->enqueue_count = q->merge_buffer_size - q->dequeue_count - q->new_root_size;
}




void *heap_queue_post_per_thread(void *arg)
{
  // printf("post_per_thread\n");

  int qid = *(int *)arg;

#ifdef HQ_PTHREADS
  cudaSetDevice(CUDA_DEVICE);
#endif

  cudaStreamSynchronize(stream[qid]);

	/* Merge enqueue buffer and FEL */
  CudaCheckError();
	merge_fel_enqueue<<<1, 1, 0, stream[qid]>>>(qid);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);
	CudaCheckError();

	bool got_elements = sort_merged_buffer(qid);
	CudaCheckError();

#if defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP) || defined(_A_STAR)
  update_counts<<<1, 1, 0, stream[qid]>>>(qid);
	CudaCheckError();
#endif

  if(!got_elements)
  {
    return NULL;
  }

#if !defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP) && !defined(_A_STAR)
#ifndef HQ_PARALLEL_PARTITION
	partition_merged<<<1, 1, 0, stream[qid]>>>(qid);
	CudaCheckError();
#else

	int h_merge_buffer_size;

  get_merge_buffer_size<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyAsync(&h_merge_buffer_size, host_pointers[qid].int_ptr, sizeof(h_merge_buffer_size), cudaMemcpyDeviceToHost, stream[qid]) );
  cudaStreamSynchronize(stream[qid]);


	int num_threads = max(NUM_LPS, h_merge_buffer_size);
	int num_blocks = ceil((float)num_threads / MAX_THREADS__HEAP_QUEUE);
	partition_merged_parallel_init<<<num_blocks, MAX_THREADS__HEAP_QUEUE, 0, stream[qid]>>>(qid);
	CudaCheckError();

	num_threads = h_merge_buffer_size;
	num_blocks = ceil((float)num_threads / MAX_THREADS__HEAP_QUEUE);
	partition_merged_parallel_min<<<num_blocks, MAX_THREADS__HEAP_QUEUE, 0, stream[qid]>>>(qid);
	CudaCheckError();

	num_threads = NUM_LPS;
	num_blocks = ceil((float)num_threads / MAX_THREADS__HEAP_QUEUE);

	partition_merged_parallel_mark<<<num_blocks, MAX_THREADS__HEAP_QUEUE, 0, stream[qid]>>>(qid);
	CudaCheckError();

	partition_merged_parallel_update_counts<<<1, 1, 0, stream[qid]>>>(qid);
	CudaCheckError();

  cudaStreamSynchronize(stream[qid]);

#endif

	sort_merged_buffer_marked(qid);
#endif

	update_dequeue<<<num_blocks_node, num_threads_node, 0, stream[qid]>>>(qid);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	update_root<<<num_blocks_node, num_threads_node, 0, stream[qid]>>>(qid);
	CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

	copy_last(qid);

	/* Process enqueue counts greater than N by iteratively starting insert i
   * processes: Initiate a new insert-update process at the root node for
   * each N newly inserted elements (i.e., if num_inserted > HQ_NODE_SIZE)
	 */

	int source_offset = 0;
	int h_enqueue_count = heap_queue_get_enqueue_count(qid);

	while (h_enqueue_count > 0) {

    int inserted_curr_iteration = min(h_enqueue_count, HQ_NODE_SIZE);
    h_enqueue_count -= inserted_curr_iteration;

    init_insert_update(qid, INSERT_FROM_MERGED_BUFFER, source_offset);

    if (h_enqueue_count > 0) {
      process_heap(qid);
      cudaStreamSynchronize(stream[qid]);

      source_offset += inserted_curr_iteration;
    }
  }

  process_heap(qid);

  cudaStreamSynchronize(stream[qid]);
  // heap_queue_check_heap(qid);

  return NULL;
}

void heap_queue_post()
{
  cudaDeviceSynchronize();

  int thread_args[HQ_NUM_QUEUES];
#ifdef HQ_PTHREADS
  pthread_t threads[HQ_NUM_QUEUES];

  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {
    thread_args[qid] = qid;
#if defined(_A_STAR) && defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
    if (queue_post_mask != NULL && queue_post_mask[qid]) {
#else
    {
#endif

      if(pthread_create(&threads[qid], NULL, heap_queue_post_per_thread, &thread_args[qid]))
      {
        printf("cannot create thread %d\n", qid);
        exit(1);
      }
    }
  }

  for (int qid = 0; qid < HQ_NUM_QUEUES; qid++) {
#if defined(_A_STAR) && defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
    if (queue_post_mask != NULL && queue_post_mask[qid]) {
#else
    {
#endif
      if(pthread_join(threads[qid], NULL)) {
        printf("cannot join thread %d\n", qid);
        exit(1);
      }
    }
  }
#else
  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {
    thread_args[qid] = qid;
#if defined(_A_STAR) && defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
    if (queue_post_mask != NULL && queue_post_mask[qid]) {
#else
    {
#endif
      heap_queue_post_per_thread(&thread_args[qid]);
    }
  }
#endif
  queue_post_mask = NULL;

  cudaDeviceSynchronize();
}

__device__ bool heap_queue_is_empty(int lp)
{
  HeapQueueData *q = &heap_queue_data[lp];
  return !(q->item_count[0] + heap_queue_length(lp));
}

#ifdef _PHOLD
long heap_queue_get_min_ts()
{
  long min_ts = LONG_MAX;
  bool all_empty = true;
  for(int qid = 0; qid < HQ_NUM_QUEUES; qid++)
  {
    int h_dequeue_count;

    get_dequeue_count<<<1, 1>>>(qid, host_pointers[qid].int_ptr);
    CudaSafeCall( cudaMemcpy(&h_dequeue_count, host_pointers[qid].int_ptr, sizeof(h_dequeue_count), cudaMemcpyDeviceToHost) );

    if (h_dequeue_count > 0) {
      all_empty = false;
      queue_item item;
      get_dequeue_buffer_zero<<<1, 1>>>(qid, host_pointers[qid].queue_item_ptr);
      CudaSafeCall( cudaMemcpy(&item, host_pointers[qid].queue_item_ptr, sizeof(queue_item), cudaMemcpyDeviceToHost) );
      min_ts = min(min_ts, item.ts);
      // printf("qid %d, min_ts: %d\n", qid, item.ts);
    }  else {
    }
  }

  return all_empty ? 0 : min_ts;
}
#endif

__device__ bool heap_queue_insert(queue_item item)
{
  int qid = item.node / (NUM_LPS / HQ_NUM_QUEUES);

  // printf("%d inserting %d/%d in %d\n", threadIdx.x, item.x, item.y, qid);
  HeapQueueData *q = &heap_queue_data[qid];

  int old_index = atomicAdd(&q->enqueue_count, 1);

  if (old_index >= ENQUEUE_BUFFER_SIZE) {
    return false;
  }

  // printf("%d inserting %d/%d\n", qid, item.x, item.y);
  q->enqueue_buffer[old_index] = item;

  return true;
}

__device__ void heap_queue_insert_or_update(queue_item item, int lp)
{
  heap_queue_insert(item);
}


#if defined(_PHOLD)
queue_item_value heap_queue_root_peek_ts()
{
  // this function is for "LAQ backed by HEAP" PHOLD only
  static_assert(HQ_NUM_QUEUES == 1, "HQ_NUM_QUEUES needs to be 1 for LAQ backed by Heap with PHOLD");
  int qid = 0;
  int root_count;

  get_item_count_zero<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpy(&root_count, host_pointers[qid].int_ptr, sizeof(int), cudaMemcpyDeviceToHost) );

  CudaCheckError();
  if(root_count == 0)
  {
    return LONG_MAX;
  }

  queue_item item;
  CudaSafeCall( cudaMemcpy((void **)&item, host_pointers[qid].host_fel_ptr, sizeof(queue_item), cudaMemcpyDeviceToHost) );
  CudaCheckError();


  return item.ts;
}
#endif

__device__ int heap_queue_peek(queue_item **item, int pos, int target_qid)
{
   int idx = threadIdx.x + blockDim.x * blockIdx.x;
#if !defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
#if !defined(_A_STAR)
  int qid = pos / HQ_NODE_SIZE;
  idx = pos % HQ_NODE_SIZE;
#else
  int qid = blockIdx.x;
#endif
#elif defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
  int qid = idx;
#else
  int qid = 0;
#endif

  if(target_qid != -1)
    qid = target_qid;

  HeapQueueData *q = &heap_queue_data[qid];
#if defined(_PHOLD)
  int item_pos = idx;
#elif defined(_A_STAR) && defined(_LOCAL_ARRAY_QUEUE_BACKED_BY_HEAP)
  int item_pos = pos;
#else
  int item_pos = q->peek_index;
#endif


  if (item_pos < q->dequeue_count) {
    *item = &(q->dequeue_buffer[item_pos]);
    // printf("returning item %d/%d\n", (*item)->x, (*item)->y);
    return item_pos;
  }

  return -1;
}

__device__ void heap_queue_set_done(int index)
{
#ifdef _A_STAR
  HeapQueueData *q = &heap_queue_data[blockIdx.x];
  q->peek_index++;
#endif
}

#ifdef _PHOLD

static __global__ void d_queue_check_phold(int qid)
{
  HeapQueueData *q = &heap_queue_data[qid];
  q->event_sum = 0;

  q->event_sum += q->dequeue_count;

  for (int i = 0; i < HQ_HEAP_SIZE; ++i) {
    q->event_sum += q->item_count[i];
  }

  for (int i = 0; i < NUM_INSERT_PROCESSES; ++i) {
    if (q->insert_process[i].current_node != -1) {
      q->event_sum += q->insert_process[i].size;
    }
  }
}

void heap_queue_check_phold()
{
  assert(HQ_NUM_QUEUES == 1);
  int qid = 0;

  d_queue_check_phold<<<1, 1, 0, stream[qid]>>>(qid);
  CudaCheckError();
  cudaStreamSynchronize(stream[qid]);

  int sum = 0;

  get_event_sum<<<1, 1, 0, stream[qid]>>>(qid, host_pointers[qid].int_ptr);
  CudaSafeCall( cudaMemcpyFromSymbol(&sum, host_pointers[qid].int_ptr, sizeof(int)) );

  if (sum != PHOLD_POPULATION * NUM_LPS) {
    printf("[ HQ ] [ PHOLD ] Error: event sum (= %d) != phold population (= %d)!\n",
        sum, PHOLD_POPULATION * NUM_LPS);
    exit(-3);
  }
}
#endif // #ifdef _PHOLD

#endif // #ifdef _HEAP_QUEUE
