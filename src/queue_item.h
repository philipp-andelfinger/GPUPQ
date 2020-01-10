#pragma once
#include <assert.h>
#include "config.h"
#include <cuda_fp16.h>
#include <stdio.h>

#ifdef COUNT_COMPARISONS
  extern __device__ unsigned long long num_comparisons;
#endif

#ifdef _PHOLD

struct event {
  long ts;
  int node;
};

typedef event queue_item;
typedef long queue_item_value;


__device__ inline bool operator<(const event& lhs, const event& rhs) {
#ifdef COUNT_COMPARISONS
  atomicAdd(&num_comparisons, 1);
  // printf("comparing %ld and %ld\n", lhs.ts, rhs.ts);
#endif
  return lhs.ts < rhs.ts;
}

__host__ __device__ inline bool operator==(const event& lhs, const event& rhs) {
  return lhs.ts == rhs.ts;
}

#endif // #ifdef _PHOLD

#ifdef _A_STAR

__device__ float h(int x0, int y0, int x1, int y1);
__device__ float he(int x0, int y0, int x1, int y1);

struct vertex {
  short x, y;
  // estimated cost of getting from start to goal through this vertex
  half f;
  unsigned short node;
  struct vertex *pred;

  static const unsigned int MAX_ID = A_STAR_GRID_X * A_STAR_GRID_Y;
};

typedef vertex queue_item;
typedef half queue_item_value;

extern __device__ struct vertex current_goal[NUM_CONCURRENT_AGENTS];

__host__ __device__ inline unsigned int item_id(short x, short y) {
  assert(x >= 0);
  assert(y >= 0);
  return y * A_STAR_GRID_X + x;
}

__host__ __device__ inline unsigned int item_id(const vertex& v) {
  return item_id(v.x, v.y);
}

__device__ inline bool operator<(const vertex& lhs, const vertex& rhs) {
#ifdef COUNT_COMPARISONS
  unsigned long long old = atomicAdd(&num_comparisons, (unsigned long long)1);
#endif

   float lf = __half2float(lhs.f);
   float rf = __half2float(rhs.f);


   if(lf == rf)
   {
     struct vertex goal = current_goal[lhs.node];
     return h(lhs.x, lhs.y, goal.x, goal.y) < h(rhs.x, rhs.y, goal.x, goal.y);
   }

   return lf < rf;
}

__host__ __device__ inline bool operator==(const vertex& lhs, const vertex& rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}


#endif // #ifdef _A_STAR


#define ITEM_BYTES ((int)sizeof(queue_item))


__device__ inline bool operator>(const queue_item& lhs, const queue_item& rhs) {
  return rhs < lhs;
}

__device__ inline bool operator<=(const queue_item& lhs, const queue_item& rhs) {
  return !(lhs > rhs);
}

__device__ inline bool operator>=(const queue_item& lhs, const queue_item& rhs) {
  return !(lhs < rhs);
}
