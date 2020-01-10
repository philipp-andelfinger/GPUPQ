/*
                An implementation of top-down splaying
                    D. Sleator <sleator@cs.cmu.edu>
    	                     March 1992

  "Splay trees", or "self-adjusting search trees" are a simple and
  efficient data structure for storing an ordered set.  The data
  structure consists of a binary tree, without parent pointers, and no
  additional fields.  It allows searching, insertion, deletion,
  deletemin, deletemax, splitting, joining, and many other operations,
  all with amortized logarithmic performance.  Since the trees adapt to
  the sequence of requests, their performance on real access patterns is
  typically even better.  Splay trees are described in a number of texts
  and papers [1,2,3,4,5].

  The code here is adapted from simple top-down splay, at the bottom of
  page 669 of [3].  It can be obtained via anonymous ftp from
  spade.pc.cs.cmu.edu in directory /usr/sleator/public.

  The chief modification here is that the splay operation works even if the
  item being splayed is not in the tree, and even if the tree root of the
  tree is NULL.  So the line:

                              t = splay(i, t);

  causes it to search for item with key i in the tree rooted at t.  If it's
  there, it is splayed to the root.  If it isn't there, then the node put
  at the root is the last one before NULL that would have been reached in a
  normal binary search for i.  (It's a neighbor of i in the tree.)  This
  allows many other operations to be easily implemented, as shown below.

  [1] "Fundamentals of data structures in C", Horowitz, Sahni,
       and Anderson-Freed, Computer Science Press, pp 542-547.
  [2] "Data Structures and Their Algorithms", Lewis and Denenberg,
       Harper Collins, 1991, pp 243-251.
  [3] "Self-adjusting Binary Search Trees" Sleator and Tarjan,
       JACM Volume 32, No 3, July 1985, pp 652-686.
  [4] "Data Structure and Algorithm Analysis", Mark Weiss,
       Benjamin Cummins, 1992, pp 119-130.
  [5] "Data Structures, Algorithms, and Performance", Derick Wood,
       Addison-Wesley, 1993, pp 367-375.
*/

#include "config.h"
#ifdef _LOCAL_SPLAY_QUEUE
#include "queue_item.h"
#include "queue.cuh"
#include "local_splay_queue.cuh"

#include <stdio.h>
#include <cub/cub.cuh>


static int num_threads_lps = min(MAX_THREADS__LOCAL_ARRAY_QUEUE, NUM_LPS);
static int num_blocks_lps = (NUM_LPS + num_threads_lps - 1) / num_threads_lps;

__device__ long global_min;
__device__ long lp_min_ts[NUM_LPS];

__device__ queue_item *fel;

__device__ int malloc_buf_pos[NUM_LPS];
__device__ int insert_count[NUM_LPS];

__device__ int size[NUM_LPS];            /* number of nodes in the tree */
           /* Not actually needed for any of the operations */

// typedef struct tree_node Tree;
__device__ tree_node *splay_root[NUM_LPS];
__device__ tree_node *malloc_buf;

__device__ tree_node *malloc_()
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // printf("malloc for idx %d\n", idx);

  for(int i = 0; i < MALLOC_BUF_SIZE; i++)
  {
    int offset = (i + malloc_buf_pos[idx]) % MALLOC_BUF_SIZE;
    int abs_offset = idx * MALLOC_BUF_SIZE + offset;
#ifdef _PHOLD
    if(malloc_buf[abs_offset].item.ts == -1)
#else
    if(__half2float(malloc_buf[abs_offset].item.f) == -1.0)
#endif
    {
      malloc_buf_pos[idx] = (offset + 1) % MALLOC_BUF_SIZE;
      return &malloc_buf[abs_offset];
    }
  }

  return NULL;
}

__device__ void free_(tree_node *node)
{
#ifdef _PHOLD
  node->item.ts = -1;
#else
  queue_item item;
  item.f = __float2half(-1.0);
  node->item = item;
#endif
}



__device__ tree_node *splay(queue_item i, tree_node * t)
{
/* Simple top down splay, not requiring i to be in the tree t.  */
/* What it does is described above.                             */
  tree_node N, *l, *r, *y;
  if (t == NULL)
    return t;
  N.left = N.right = NULL;
  l = r = &N;

  for (;;) {
    if (i <= t->item) {
      if (t->left == NULL)
        break;
      if (i < t->left->item) {
        y = t->left;            /* rotate right */
        t->left = y->right;
        y->right = t;
        t = y;
        if (t->left == NULL)
          break;
      }
      r->left = t;              /* link right */
      r = t;
      t = t->left;
    } else if (i > t->item) {
      if (t->right == NULL)
        break;
      if (i > t->right->item) {
        y = t->right;           /* rotate left */
        t->right = y->left;
        y->left = t;
        t = y;
        if (t->right == NULL)
          break;
      }
      l->right = t;             /* link left */
      l = t;
      t = t->right;
    } /* else {
      break;
    } */
  }
  l->right = t->left;           /* assemble */
  r->left = t->right;
  t->left = N.right;
  t->right = N.left;
  return t;
}

__device__ tree_node *insert(queue_item i, tree_node * t)
{
/* Insert i into the tree t, unless it's already there.    */
/* Return a pointer to the resulting tree.                 */

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  tree_node *new_;

  // new_ = (tree_node *) malloc(sizeof(tree_node));
  new_ = (tree_node *) malloc_();
  // printf("%d: new is %p\n", threadIdx.x, new_);
  if (new_ == NULL) {
    printf("Ran out of space\n");
      return NULL;
  }
  new_->item = i;
  if (t == NULL) {
    new_->left = new_->right = NULL;
    size[idx] = 1;
    return new_;
  }
  t = splay(i, t);
  if (i <= t->item) {
    new_->left = t->left;
    new_->right = t;
    t->left = NULL;
    size[idx]++;
    return new_;
  } else if (i > t->item) {
    new_->right = t->right;
    new_->left = t;
    t->right = NULL;
    size[idx]++;
    return new_;
  }

  assert(false);
  return NULL;
//    else {                      /* We get here if it's already in the tree */
//     /* Don't add it again                      */
//     free_(new_);
//     return t;
//   }
}


__device__ int counted_size[NUM_LPS];
__device__ void dump_tree(tree_node *node)
{
  if(node == splay_root[threadIdx.x])
    counted_size[threadIdx.x] = 0;

  if(node == NULL)
  {
    printf("NULL\n");
    return;
  }
  counted_size[threadIdx.x]++;
  //printf("%ld\n", node->item.ts);
  //printf("%.2f\n", __half2float(node->item.f));

  printf("left: ");
  dump_tree(node->left);

  printf("right: ");
  dump_tree(node->right);

  printf("up\n");
  if(node == splay_root[threadIdx.x] && counted_size[threadIdx.x] != size[threadIdx.x])
    printf("%d: size is wrong: %d vs %d\n", threadIdx.x, counted_size[threadIdx.x], size[threadIdx.x]);
    

}

__device__ tree_node *delete_min(queue_item * min, tree_node * root, int lp)
{
/* Deletes the minimum from the tree */
/* Return a pointer to the resulting tree.              */
  // printf("%d: root is %p\n", threadIdx.x + blockIdx.x * blockDim.x, root);
  tree_node *x;
  if (root == NULL)
    return NULL;
  tree_node *t = root;
  while(t->left != NULL)
    t = t->left;

  *min = t->item;

  /* if(!(threadIdx.x + blockIdx.x * blockDim.x))
  {
    printf("before:\n");
    dump_tree(t);
  } */

  t = splay(t->item, root);
  
  /* if(!(threadIdx.x + blockIdx.x * blockDim.x))
  {
    printf("after:\n");
    dump_tree(t);
  } */
  if (t->left == NULL) {
    // printf("%d: t->left is NULL, t->right is %p\n", threadIdx.x + blockIdx.x * blockDim.x, t->right);
    x = t->right;
  } else {
    // printf("%d: t->left is not NULL, t->right is %p\n", threadIdx.x + blockIdx.x * blockDim.x, t->right);
    x = splay(t->item, t->left);
    x->right = t->right;
  }
  size[lp]--;

  free_(t);
  // printf("%d returning %p\n", threadIdx.x + blockIdx.x * blockDim.x, x);
  return x;
}

__device__ bool local_splay_queue_insert(queue_item item)
{
  int lp = get_lp(item.node);
  int insert_pos = atomicAdd(&insert_count[lp], 1);
  int index = lp * FEL_SIZE + insert_pos;

  //if (VERBOSE_DEBUG) {
/* #ifdef _PHOLD
  if(lp == 1)
    printf("inserting item with ts %ld at insert pos %d, index %d\n", item.ts,
        insert_pos, index);
#endif */
  //}

  fel[index] = item;

  return true;
}

__global__ void local_splay_queue_init_d()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < NUM_LPS; idx += blockDim.x * gridDim.x)
  {
    splay_root[idx] = NULL;
    for(int i = 0; i < MALLOC_BUF_SIZE; i++)
    {
#ifdef _PHOLD
      malloc_buf[idx * MALLOC_BUF_SIZE + i].item.ts = -1;
#else
      queue_item item;
      item.f = __float2half(-1.0);
      malloc_buf[idx * MALLOC_BUF_SIZE + i].item = item;
#endif
    }
  }
}

void local_splay_queue_init()
{
  printf("\n\n-----------------------------------\n");
  printf("[ LSQ ] Memory consumption\n");
  printf("-----------------------------------\n");
  printf(" available:     %.2f MB\n", (float) DEVICE_MEMORY_MB);
  printf(" enqueue buf:   %d MB (%d items per enqueue buffer)\n", FEL_SIZE * ITEM_BYTES * NUM_LPS / 1024 / 1024, FEL_SIZE);
  printf(" malloc buf:    %d MB (%d items per malloc buffer)\n", MALLOC_BUF_SIZE * NUM_LPS * sizeof(tree_node) / 1024 / 1024, MALLOC_BUF_SIZE);
    
  printf("-----------------------------------\n\n");


  queue_item *h_fel;
  CudaSafeCall( cudaMalloc(&h_fel, ITEM_BYTES * FEL_SIZE * NUM_NODES) );
  CudaSafeCall( cudaMemcpyToSymbol(fel, &h_fel, sizeof(fel)) );

  tree_node *h_malloc_buf;
  CudaSafeCall( cudaMalloc(&h_malloc_buf, sizeof(tree_node) * MALLOC_BUF_SIZE * NUM_NODES) );
  CudaSafeCall( cudaMemcpyToSymbol(malloc_buf, &h_malloc_buf, sizeof(malloc_buf)) );

  local_splay_queue_init_d<<<num_blocks_lps, num_threads_lps>>>();
  cudaDeviceSynchronize();

}

__device__ int queue_peek(queue_item **item, int lp)
{
  // return and splay min
  tree_node *root = splay_root[lp];
  if (root == NULL)
  {
    *item = NULL;
    return -1;
  }
  tree_node *t = root;
  while(t->left != NULL)
    t = t->left;
#ifdef _PHOLD
  if(t->item.ts >= global_min + LOOKAHEAD)
  {
    *item = NULL;
    return -1;
  }
#endif
  *item = &(t->item);
  t = splay(t->item, root);
  splay_root[lp] = t;
  return lp;
}

__device__ void local_splay_queue_set_done(int index)
{
  int lp = index;
  tree_node *root = splay_root[lp];
  queue_item item;
  splay_root[lp] = delete_min(&item, root, lp);
  // dump_tree(splay_root[lp]);
}

void local_splay_queue_finish()
{
}

#ifdef _PHOLD
static __global__ void find_min_ts_device_pre()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {
    queue_item *item;
    queue_peek(&item, idx);
    if (item != NULL) {
      lp_min_ts[idx] = item->ts;
    } else {
      lp_min_ts[idx] = LONG_MAX;
    }
  }
}
#endif

#ifdef _PHOLD
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

#ifdef _PHOLD
long local_splay_queue_get_min_ts()
{
  long dummy_ts = LONG_MAX - LOOKAHEAD;
  CudaSafeCall( cudaMemcpyToSymbol(global_min, &dummy_ts, sizeof(long)) );

  find_min_ts_device_pre<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

  find_min_ts_device<<<1, 1>>>();
  CudaCheckError();

  long min;

  CudaSafeCall( cudaMemcpyFromSymbol(&min, global_min, sizeof(long)) );

  return min;
}
#endif

__global__ void insert_bulk()
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < NUM_LPS;
      idx += blockDim.x * gridDim.x) {

    tree_node *root = splay_root[idx];


    for(int i = 0; i < insert_count[idx]; i++)
    {
      root = insert(fel[idx * FEL_SIZE + i], root);
    }
    splay_root[idx] = root;
    
    insert_count[idx] = 0;


  }
}

__device__ void clear_tree(int lp, tree_node *node)
{
  if(node == NULL)
  {
    return;
  }

  clear_tree(lp, node->left);
  clear_tree(lp, node->right);

  free_(node);
}

__global__ void queue_clear_(int lp)
{
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < MALLOC_BUF_SIZE;
      idx += blockDim.x * gridDim.x)
  {
#ifdef _PHOLD
    malloc_buf[lp * MALLOC_BUF_SIZE + idx].item.ts = -1;
#else
    queue_item item;
    item.f = __float2half(-1.0);
    malloc_buf[lp * MALLOC_BUF_SIZE + idx].item = item;
#endif
  }
}

__device__ void queue_clear(int lp)
{
  
  queue_clear_<<<MALLOC_BUF_SIZE / 256 > 0 ? MALLOC_BUF_SIZE / 256 : 1, 256>>>(lp);

  
  splay_root[lp] = NULL;
  size[lp] = 0;
  insert_count[lp] = 0;
}

void local_splay_queue_pre()
{
}


void local_splay_queue_post()
{
  insert_bulk<<<num_blocks_lps, num_threads_lps>>>();
  CudaCheckError();

}

void local_splay_queue_post_init()
{
  local_splay_queue_post();
}

__device__ bool queue_is_empty(int lp)
{
  return size[lp] + insert_count[lp] == 0;
}

__device__ int queue_length(int lp)
{
  return size[lp];
}

__device__ void queue_insert_or_update(queue_item item, int lp)
{
  queue_insert(item);
}
#endif
