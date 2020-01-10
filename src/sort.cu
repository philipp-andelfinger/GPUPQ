#include "config.h"
#include "queue.cuh"

#include "sort.cuh"

#include <stdio.h>

#ifndef FEL_SIZE
#define FEL_SIZE 1
#endif

/* perform i % j while checking for a negative result and wrapping it around the
 * end:
 *   wrap(5, 4) = 1
 *   wrap(-1, 4) = 3
 */
__device__ int wrap(int i)
{
  int result = i % FEL_SIZE;

  if (result >= 0) {
    return result;
  }

  return FEL_SIZE + result;
}

/* Finds the index of <item> in the range [start, end) of <ringbuffer> with <capacity>.
 * <start> is allowed to be larger than <end>, i.e. the range are allowed to cross the end of the ringbuffer.
 * Comparison is done with operator==. Returns -1 if the element is not found.
 */
__device__ int find_in_ringbuffer(queue_item *ringbuffer, int capacity, int start, int end, queue_item item) {
  assert(start >= 0);
  assert(end >= 0);
  assert(capacity >= 0);
  assert(ringbuffer != nullptr);

  for (int i = start; i != end; i = (i + 1) % capacity) {
    if (ringbuffer[i] == item) {
      return i;
    }
  }
  return -1;
}

/* search for the right insert position for <item> within [read, write) in the
 * ringbuffer (i.e., fel[lp * FEL_SIZE]), comparing with operator>. */
__device__ int search_ringbuffer(queue_item *ringbuffer, int read, int write, queue_item item)
{
  int start = (read - 1 + FEL_SIZE) % FEL_SIZE;
  int end = (write + 1) % FEL_SIZE;
  int i;

  for (i = start; i != end; i = (i + 1) % FEL_SIZE)
  {
    int next = (i + 1) % FEL_SIZE;

    if (i + 1 == write) {
      return i + 1;
    } else if (ringbuffer[next] > item) {
      return i;
    }
  }

  return i;
}

static __device__ void swap(queue_item *data, int x, int y)
{
  queue_item temp = data[x];
  data[x] = data[y];
  data[y] = temp;
}

static __device__ int partition(queue_item *data, int lp, int left, int right)
{
  const int mid = left + (right - left) / 2;

  const queue_item pivot = data[wrap(mid)];

  swap(data, wrap(mid), wrap(left));

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && data[wrap(i)] <= pivot) {
      i++;
    }

    while (i <= j && data[wrap(j)] > pivot) {
      j--;
    }

    if (i < j) {
      swap(data, wrap(i), wrap(j));
    }
  }

  swap(data, wrap(i - 1), wrap(left));
  return i - 1;
}


__device__ void quicksort_seq(queue_item *data, int lp, int left, int right)
{
  if(left == right)
    return;

  if (left > right) {
    right = FEL_SIZE + right;
  }

  int stack_size = 0;
  sort_data stack[QUICKSORT_STACK_SIZE];

  stack[stack_size++] = {left, right};

  while (stack_size > 0) {

    int curr_left = stack[stack_size - 1].left;
    int curr_right = stack[stack_size - 1].right;
    stack_size--;

    if (curr_left < curr_right) {
      int part = partition(data, lp, curr_left, curr_right);
      stack[stack_size++] = {curr_left, part - 1};
      stack[stack_size++] = {part + 1, curr_right};
    }
  }
}

__device__ void insert_sorted_parallel(queue_item *data, int read, int write, queue_item item)
{
  const unsigned int tIdx = threadIdx.x;
  __shared__ bool finished;

  if(!tIdx)
    finished = false;

  if(write < read)
    write += FEL_SIZE;
  write -= read;

  int i;

  int nBlockDim = -blockDim.x;
  bool insert = false;

  int insert_pos;
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
  for(i = write - blockDim.x + 1; i > nBlockDim + 1; i -= blockDim.x)
#else
  for(i = -1; i < write - 1; i += blockDim.x)
#endif
  {
    bool copy = false;

    queue_item tmp = {-1, -1}; // must be smaller than any real queue_item

#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
    if(i + (int)tIdx > 0)
      tmp = data[wrap(read + i + tIdx - 1)];
#else
    if(i + (int)tIdx < write - 1)
      tmp = data[wrap(read + i + tIdx + 1)];
    /* if(tmp.x != -1)
      printf("checking %d/%d %.2f\n", tmp.x, tmp.y, __half2float(tmp.f)); */
#endif

    __syncthreads();
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
    if(i + (int)tIdx > 0)
#else
    if(i + (int)tIdx < write - 1)
#endif
    {
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
      if(tmp > item)
#else
      if(tmp < item)
#endif
      {
        copy = true;
      }
      else
      {
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
        if(i + (int)tIdx == write || data[wrap(read + i + tIdx)] > item)
#else
        if(i + (int)tIdx == -1 || data[wrap(read + i + tIdx)] < item)
#endif
        {
          insert = true;
          insert_pos = i + (int)tIdx;
        }
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
        if(tIdx == blockDim.x - 1)
#else
        if(tIdx == 0)
#endif
          finished = true;
      }
      if(copy)
      {
        data[wrap(read + i + tIdx)] = tmp;
      }
    }
    __syncthreads(); // for element at block border and 'finished'
    if(finished)
      break;
  }

#if !defined(LOCAL_ARRAY_QUEUE_BACKWARD)
  if(!tIdx && (data[wrap(read + write - 1)] < item))
  {
    insert = true;
    insert_pos = write - 1;
  }
#endif

  if(!tIdx && ((write == 0) || data[wrap(read + 0)] > item))
  {
    // printf("%d: c\n", tIdx + i);

    insert = true;
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
    insert_pos = 0;
#else
    insert_pos = -1;
#endif
  }

  __syncthreads();

  if(insert)
  {
    data[wrap(read + insert_pos)] = item;
  }


}

__device__ int insertion_buf_wrap(int i)
{
  int result = i % LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE;

  if (result >= 0) {
    return result;
  }

  return LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE + result;
}


/* perform an insertion sort with <ev> into <data> in the range [read, write) */
__device__ void insert_sorted_insertion_buf(queue_item *data, int read, int write, queue_item item)
{
  int start = read;
  int end = insertion_buf_wrap(write);

  if (read == write) {
    data[insertion_buf_wrap(read - 1)] = item;
    return;
  }

  int prev = insertion_buf_wrap(start - 1);
  for (int i = start; i != end; i = insertion_buf_wrap(i + 1)) {
    prev = insertion_buf_wrap(i - 1);
    if (item > data[i]) {
      data[prev] = data[i];
    } else {
      data[prev] = item;
      return;
    }
  }

  data[insertion_buf_wrap(end - 1)] = item;
}

#ifdef LOCAL_ARRAY_QUEUE_SINGLE_PASS
/* perform an insertion sort with <ev> into <data> in the range [read, write) */
__device__ void insert_sorted_single_pass(queue_item *data, int read, int write, int num_new)
{
  if(num_new == 0)
    return;

  if(num_new > LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE)
  {
    printf("LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE exceeded\n");
    return;
  }

  __shared__ queue_item insertion_buf_[MAX_THREADS__LOCAL_ARRAY_QUEUE * LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE];
  queue_item *insertion_buf = &insertion_buf_[threadIdx.x * LOCAL_ARRAY_QUEUE_SINGLE_PASS_MAX_ENQUEUE];

  int insertion_buf_read = 0;
  int insertion_buf_write = 0;
  for(int i = 0; i < num_new; i++)
  {
    queue_item item = data[wrap(write + i)];

    insert_sorted_insertion_buf(&insertion_buf[0], insertion_buf_read, 0, item);
    insertion_buf_read = insertion_buf_wrap(insertion_buf_read - 1);
  }

  int queue_elements = read <= write ? (write - read) : write - read + FEL_SIZE;
  for(int i = 0; i < queue_elements + num_new; i++)
  {
    if(i >= queue_elements)
    {
      queue_item i_item = insertion_buf[insertion_buf_read];
      data[wrap(read + i)] = i_item;
      insertion_buf_read = insertion_buf_wrap(insertion_buf_read + 1);
    }
    else if(insertion_buf[insertion_buf_read] <= data[wrap(read + i)])
    {
      queue_item i_item = insertion_buf[insertion_buf_read];
      insertion_buf_read = insertion_buf_wrap(insertion_buf_read + 1);
      insert_sorted_insertion_buf(&insertion_buf[0], insertion_buf_read, insertion_buf_write, data[wrap(read + i)]);
      insertion_buf_read = insertion_buf_wrap(insertion_buf_read - 1);
      data[wrap(read + i)] = i_item;
    }
  }
}
#endif

/* perform an insertion sort with <ev> into <data> in the range [read, write) */
__device__ void insert_sorted(queue_item *data, int read, int write, queue_item item)
{
#ifdef LOCAL_ARRAY_QUEUE_BACKWARD
  int start = wrap(write - 1);
  int end = wrap(read - 1);

  int queue_elements = read <= write ? (write - read) : write - read + FEL_SIZE;

  if (read == write || data[start] <= item) {
    data[write] = item;
    return;
  }

  for (int i = start; i != end; i = wrap(i - 1)) {
    if (data[i] > item) {
      int next = wrap(i + 1);
      data[next] = data[i];
    }

    int prev = wrap(i - 1);
    if (i == read || data[prev] <= item) {
      data[i] = item;
      return;
    }
  }
#else

  int start = wrap(read);
  int end = wrap(write);
  int final = wrap(write - 1);

  int queue_elements = read <= write ? (write - read) : write - read + FEL_SIZE;

  if (read == write || data[start] >= item) {
    data[wrap(start - 1)] = item;
    return;
  }

  for (int i = start; i != end; i = wrap(i + 1)) {
    if (data[i] < item) {
      int prev = wrap(i - 1);
      data[prev] = data[i];
    }

    int next = wrap(i + 1);
    if (i == final || data[next] >= item) {
      data[i] = item;
      return;
    }
  }
#endif
}
