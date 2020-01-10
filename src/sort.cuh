#include "queue_item.h"

typedef struct sort_data {
  int left;
  int right;
} sort_data;

__device__ void quicksort_seq(queue_item *data, int lp, int left, int right);

__device__ int find_in_ringbuffer(queue_item *ringbuffer, int capacity, int start, int end, queue_item item);

__device__ int search_ringbuffer(queue_item *ringbuffer, int read, int write, queue_item item);

__device__ void insert_sorted(queue_item *data, int read, int write, queue_item item);

__device__ int wrap(int i, int j);

__device__ void insert_sorted_parallel(queue_item *data, int read, int write, queue_item item);

__device__ void insert_sorted_single_pass(queue_item *data, int read, int write, int num_new);
