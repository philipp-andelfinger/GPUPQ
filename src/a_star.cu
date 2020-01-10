#include "config.h"

#ifdef _A_STAR

#include <assert.h>

#include "model.h"
#include "queue.cuh"
#include "queue.h"
#include "util.cuh"


#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "cuda_bitset.cuh"

#include <cuda_fp16.h>

#include "gl_interop.cuh"

// Execution Model:
// a CUDA block corresponds to a single agent
// blocks are one-dimensional and consist of ACTIVE_THREADS_PER_CLI * A_STAR_MAX_PARALLEL_OL_ENTRIES threads
// the CUDA grid is one-dimensional and consists of NUM_CONCURRENT_AGENTS blocks

// Having more than one node assigned per LP is not useful, as their f value might be completely different.
static_assert(NUM_NODES_PER_LP == 1, "A* does not work with more than one node per lp");

// the following settings depending on the dimensionality of the vertex:
// distance to explore new positions from the center
#define EXPLORE_RADIUS 1
// diameter of the square around the center which gets explored
#define EXPLORE_DIAMETER (2 * EXPLORE_RADIUS + 1)
// number of threads which work on the surrounding elements of a single closed list index (CLI):
// one for each position in a square around the center, - 1 for the center itself.
#define ACTIVE_THREADS_PER_CLI (EXPLORE_DIAMETER * EXPLORE_DIAMETER - 1)

// Assume worst case for closed list: All positions in the grid get visited
#define MAX_CL_BUFFER_SIZE (A_STAR_GRID_X * A_STAR_GRID_Y / 2)

static_assert(NUM_AGENTS > 0, "A* requires the number of agents to run (MAX_EVENTS) to be > 0.");

#define NUM_THREADS_PER_AGENT (ACTIVE_THREADS_PER_CLI * A_STAR_MAX_PARALLEL_OL_ENTRIES)

// Assume a cycle if path length > CYCLE_THRESHOLD, and stop backtracing then
#define CYCLE_THRESHOLD (A_STAR_GRID_X * A_STAR_GRID_Y)


int *h_map = NULL;
struct vertex *h_starts = NULL;
struct vertex *h_goals  = NULL;
cudaTextureObject_t texObj = 0;


// points to the next agent that is not currently executed by a block.
__device__ int next_sg_agent = 0;
// number of agents that were successfuly executed
__device__ int completed_sg_agents = 0;
// number of agents for which a valid path was found
__device__ int sg_agents_with_path = 0;
// linearized map used to draw the found paths for the agents
__device__ int out_map[A_STAR_GRID_X * A_STAR_GRID_Y];


//
// open list priority queue
//
// values of the cost estimation function f. For each coordinate not in the open list, this is set to 0.
// This caches the f values stored in the vertices of the open list, to be able to access them via coordinates.
// __device__ float f_ol[NUM_CONCURRENT_AGENTS][A_STAR_GRID_Y][A_STAR_GRID_X];
__device__ half f_ol[NUM_CONCURRENT_AGENTS][A_STAR_GRID_Y][A_STAR_GRID_X];


//
// closed list
//
// implemented as ring buffer, cl_head is the index of the first queue element, cl_length is the length of the queue.
// cl_offset is used to work on multiple entries in parallel, regarding to thread index.
// dynamically allocate cl as it's too big for static allocation segment
#ifdef A_STAR_TRACE_PATHS
__device__ struct vertex *cl = NULL;
__device__ int cl_head[NUM_CONCURRENT_AGENTS];
__device__ int cl_length[NUM_CONCURRENT_AGENTS];
#endif

// bitmap with one bit per coordinate to cache whether a coordinate is in the closed list.
__device__ cuda_bitset<vertex::MAX_ID> *in_cl;

__device__ struct vertex starts[NUM_AGENTS];
__device__ struct vertex goals[NUM_AGENTS];

__device__ int current_agent[NUM_CONCURRENT_AGENTS];
__device__ struct vertex current_goal[NUM_CONCURRENT_AGENTS];
__device__ bool found_goal[NUM_CONCURRENT_AGENTS];

#ifdef A_STAR_CALCULATE_QUEUE_LENGTH
__device__ unsigned long long queue_length_observations;
__device__ unsigned long long queue_length_sum;
#endif

#ifdef A_STAR_REALTIME_VISUALIZATION
extern cudaSurfaceObject_t bitmap_surface;
#endif

__device__
float he(int x0, int y0, int x1, int y1)
{
  float dx = x1 - x0;
  float dy = y1 - y0;
  return sqrt(dx * dx + dy * dy);
}

__device__
float h(int x0, int y0, int x1, int y1)
{
  int dx = abs(x0 - x1);
  int dy = abs(y0 - y1);

  int min_ = min(dx, dy);
  int max_ = max(dx, dy);

  float h = sqrt(2.0) * min_ + max_ - min_;

  return h;
}

__device__
int cl_offset(int head, int i)
{
  int sum = head + i;

  if (sum < MAX_CL_BUFFER_SIZE) {
    return sum;
  }
  return sum - MAX_CL_BUFFER_SIZE;
}


__device__
static int get_new_agent() {
    __shared__ int new_sg_agent;

    if (threadIdx.x == 0) {
      new_sg_agent = atomicAdd(&next_sg_agent, 1);
      // printf("%d got agent %d\n", blockIdx.x, new_sg_agent);
    }
    __syncthreads();

    if (new_sg_agent >= NUM_AGENTS) {
      return -1;
    } else {
      return new_sg_agent;
    }
}

__device__
#ifdef A_STAR_REALTIME_VISUALIZATION
static void init_agent(int block_id, int agent_id, cudaSurfaceObject_t surface) {
#else
static void init_agent(int block_id, int agent_id) {
#endif
  current_agent[blockIdx.x] = agent_id;


  /* starts[agent_id].x = 241;
  starts[agent_id].y = 85;

  goals[agent_id].x = 234;
  goals[agent_id].y = 221; */

  vertex start = starts[agent_id];
  vertex goal = current_goal[blockIdx.x] = goals[agent_id];

  // start.f = __float2half(0.0); // __float2half(h(start.x, start.y, goal.x, goal.y));
  start.f = __float2half(h(start.x, start.y, goal.x, goal.y));
  start.node = block_id;


  //if (VERBOSE_DEBUG && threadIdx.x == 0) {
    //printf("New agent %d for [%d]: (%d, %d) -> (%d, %d)\n", agent_id, block_id, start.x, start.y, goal.x, goal.y);
  //}

  if (threadIdx.x == 0) {
    // reset everything to be able to simulate a new agent

    //TIMER_TIC

    for (int i = 0; i < A_STAR_GRID_Y; i++) {
      for (int j = 0; j < A_STAR_GRID_X; j++) {
        f_ol[block_id][i][j] = __float2half(0.0);
      }
    }

    // reset in_cl to 0 to execute a new agent
    in_cl[block_id].clear();

    found_goal[block_id] = false;

#ifdef A_STAR_TRACE_PATHS
    cl_head[block_id]   = 0;
    cl_length[block_id] = 0;
#endif

    // printf("%d, %d: calling queue_insert\n", block_id, agent_id);
    queue_insert(start);
    //TIMER_TOC
#ifdef A_STAR_REALTIME_VISUALIZATION
    int x = goal.x;
    int y = goal.y;

    uchar4 pixel = { 255, (unsigned char)(agent_id * 1234), (unsigned char)(agent_id * 123), 255 };
    for(int r = 0; r < 7; r++)
    {
      for(int w = 0; w < 2; w++)
      {
        if(x - r + w < 0 || x - r + w >= A_STAR_GRID_X ||
           x + r + w < 0 || x + r + w >= A_STAR_GRID_X ||
           y - r < 0 || y - r >= A_STAR_GRID_Y ||
           y + r < 0 || y + r >= A_STAR_GRID_Y)
          continue;
        surf2Dwrite(pixel, surface, (x - r + w) * sizeof(uchar4), y - r);
        surf2Dwrite(pixel, surface, (x - r + w) * sizeof(uchar4), y + r);
        surf2Dwrite(pixel, surface, (x + r + w) * sizeof(uchar4), y - r);
        surf2Dwrite(pixel, surface, (x + r + w) * sizeof(uchar4), y + r);
      }
    }
#endif


  }
  __syncthreads();
}

#ifdef A_STAR_REALTIME_VISUALIZATION
__global__
void redraw_surface(cudaTextureObject_t texObj, cudaSurfaceObject_t surface)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= A_STAR_GRID_X)
    return;

  uchar4 pixel_clear = { 0, 0, 0, 255};
  uchar4 pixel_obstacle = { 50, 50, 50, 255 };

  if(!tex2D<int>(texObj, x, y))
    surf2Dwrite(pixel_clear, surface, x * sizeof(uchar4), y);
  else
    surf2Dwrite(pixel_obstacle, surface, x * sizeof(uchar4), y);
}
#endif

// States in which this gets called:
// 1. Initial Call => Fetch new agent, return
// 2. There's an agent for the block, whose ol_length is != 0 => dequeue ol, look at it, put the results in ol, return
// 3. There's an agent for the block, whose ol_length is == 0 => if a way has been found: trace back and store, if not,
//                                                               print that instead. Then fetch new agent, return.
__global__
#ifdef A_STAR_REALTIME_VISUALIZATION
static void a_star(cudaTextureObject_t texObj, cudaSurfaceObject_t surface)
#else
static void a_star(cudaTextureObject_t texObj)
#endif
{
  const unsigned int tIdx = threadIdx.x;
  // the index of the agent this code will ecxecute, out of the list of all currently executed agents.
  int agent = blockIdx.x;

  if (tIdx == 0) {
    assert(texObj != NULL);
    assert(in_cl  != NULL);
  }
  // determine the offsets for this agent's cl
#ifdef A_STAR_TRACE_PATHS
  struct vertex *local_cl = cl + agent * MAX_CL_BUFFER_SIZE;
#else
  __shared__ struct vertex local_cl[A_STAR_MAX_PARALLEL_OL_ENTRIES];
#endif


  // TIMER_INIT
  // TIMER_TIC

  if (queue_is_empty(agent)) {
    // the last agent has ben finished, get a new agent to execute

    // the index of the agent this code will execute, out of the list of all available agents.
    // this should be stored between executions, but is only needed for printf
    int sg_agent = get_new_agent();

    if (sg_agent == -1) {
      // all agents have been executed
      return;
    }

    /* if(!threadIdx.x)
      printf("queue empty, init_agent()\n"); */
#ifdef A_STAR_REALTIME_VISUALIZATION
    init_agent(agent, sg_agent, surface);
#else
    init_agent(agent, sg_agent);
#endif
  }
  else {
    // continue to evaluate the current agent

    // collect new ol vertices to insert all at once after computing them.
    // each thread only looks at one candidate for the ol, so this is enough.
    __shared__ struct vertex ol_insert_list[NUM_THREADS_PER_AGENT];

    int sg_agent = current_agent[agent];
    struct vertex goal = goals[sg_agent];

    int numParallelOlEntries = min(queue_length(agent), A_STAR_MAX_PARALLEL_OL_ENTRIES);
    __syncthreads();

    // take numParallelOlEntries new current vertices from the open list, and add them to the closed list
    if (tIdx == 0) {

      // printf("next iteration: %d\n", numParallelOlEntries);
      int num_new = 0;
      for (int i = 0; i < numParallelOlEntries; i++) {

        // remove a vertex from the open list

        struct vertex *new_vertex = NULL;
        int vertex_index = queue_peek(&new_vertex, agent);

        if (new_vertex == NULL) {
          if (queue_is_empty(agent)) {
            break;
          }
          continue;
        }
        // printf("vertex_index: %d\n", vertex_index);

        // add current vertex to the closed list
        // adding MAX_CL_BUFFERSIZE is necessary to make the result positivein the case of cl_head[agent] being 0.
#ifdef A_STAR_TRACE_PATHS
        cl_head[agent] = (cl_head[agent] - 1 + MAX_CL_BUFFER_SIZE) % MAX_CL_BUFFER_SIZE;
        cl_length[agent] = cl_length[agent] + 1;
        local_cl[cl_offset(cl_head[agent], 0)] = *new_vertex;
#else
        local_cl[num_new] = *new_vertex;
#endif

        in_cl[agent].set(item_id(*new_vertex), true);

#ifdef A_STAR_WRITE_IMAGES
        out_map[new_vertex->y * A_STAR_GRID_X + new_vertex->x] = sg_agent + 5;
#endif

#if defined(A_STAR_REALTIME_VISUALIZATION) && !defined(A_STAR_REALTIME_VISUALIZATION_PATHS_ONLY)
        uchar4 pixel = { 255 / 2, (unsigned char)(sg_agent * 1234) / 2, (unsigned char)(sg_agent * 123) / 2, 255 };

        surf2Dwrite(pixel, surface, new_vertex->x * sizeof(uchar4), new_vertex->y);
#endif

        queue_set_done(vertex_index);

        num_new++;
      }
    }


    // index of the cl entry which is assigned to the current thread
    int cl_index = tIdx / ACTIVE_THREADS_PER_CLI;

    __syncthreads();


    struct vertex *curr_vertex = NULL;
    // distribute the newly-added vertices from the ol to the threads
    if (tIdx < NUM_THREADS_PER_AGENT && cl_index < numParallelOlEntries) {
#ifdef A_STAR_TRACE_PATHS
      curr_vertex = &local_cl[cl_offset(cl_head[agent], cl_index)];
#else
      curr_vertex = &local_cl[cl_index];
#endif
    }

    int new_x, new_y;
    // compute coordinates visit: the area of EXPLORE_DIAMETER around the current vertex
    if (curr_vertex != NULL) {
      // the index of this thread inside the set of threads which evaluate curr_vertex
      int cltIdx = tIdx % ACTIVE_THREADS_PER_CLI;
      // skip the center position of the explored square as it's already visited
      int lCltIdx = cltIdx >= ACTIVE_THREADS_PER_CLI / 2 ? cltIdx + 1 : cltIdx;

      // force coordinate to be between 0 and A_STAR_GRID_X - 1
      new_x = curr_vertex->x - EXPLORE_RADIUS + lCltIdx % EXPLORE_DIAMETER;
      new_x = max(0, min(A_STAR_GRID_X - 1, new_x));
      new_y = curr_vertex->y - EXPLORE_RADIUS + lCltIdx / EXPLORE_DIAMETER;
      new_y = max(0, min(A_STAR_GRID_Y - 1, new_y));

      if (cltIdx == 0) {
        // One of the coordinates around curr_vertex is actually the goal! Stop execution.
        if (curr_vertex->x == goal.x && curr_vertex->y == goal.y) {
          // printf("ql: %d\n", queue_length(agent));
          /* if (VERBOSE)  { */
            //if(!agent)
              // printf("%d, %d: found shortest path, cost: %.2f, closed list size: %d, head: %d\n", agent, sg_agent, __half2float(curr_vertex->f), cl_length[agent], cl_head[agent]);
              // printf("%d, %d: found shortest path, cost: %.2f\n", agent, sg_agent, __half2float(curr_vertex->f));
          // }

#ifdef A_STAR_TRACE_PATHS
          // draw the found path to the out_map, color-coded by which agent was executed.
          struct vertex *pred = curr_vertex;
          int path_length = 0;
          do {
            out_map[pred->y * A_STAR_GRID_X + pred->x] = 3 + agent;

#if defined(A_STAR_REALTIME_VISUALIZATION)
            uchar4 pixel = { 255, (unsigned char)(sg_agent * 1234), (unsigned char)(sg_agent * 123), 255 };
            surf2Dwrite(pixel, surface, pred->x * sizeof(uchar4), pred->y);
#endif

            if(pred->pred != NULL && (abs(pred->pred->x - pred->x) > 1 || abs(pred->pred->y - pred->y) > 1))
              printf("%d/%d cannot be a neighbor of %d/%d...\n", pred->pred->x, pred->pred->y, pred->x, pred->y);
            pred = pred->pred;
            if (path_length > CYCLE_THRESHOLD) {
              printf("%d: probable pred pointer cycle (%d/%d -> %d/%d)\n", sg_agent, starts[agent].x, starts[agent].y, goals[agent].x, goals[agent].y);
              break;
            }
            path_length++;
          } while (pred != NULL);
#endif

          found_goal[agent] = true;
          queue_clear(agent);
        }
      }
    }
    __syncthreads();

    // mark all insert list entries as empty
    if (tIdx < NUM_THREADS_PER_AGENT) {
      ol_insert_list[tIdx].x = -1;
    }

    // skip looking at new vertices and their insertion if goal was already found
    if (!found_goal[agent]) {
      // if the computed coordinates are neither a wall or were already visited, add them to the open list
      if (curr_vertex != NULL) {

        // check whether the new coordinates are either in the closed list and therefore already investigated
        // or are a wall and therefore unpassable
        if (!in_cl[agent].get(item_id(new_x, new_y)) && !tex2D<int>(texObj, new_x, new_y)) {
          // A* is memory bound for us, so we try to save memory:
          // Unlike as in the usual A* implementation, we don't store the known distance from start to vertex inside the
          // vertex as g, but recompute g from the stored f value

          // current f - estimation from curr_vertex to goal = known cost from start to curr_vertex
          // known cost from start to curr_vertex + distance from curr_vertex to new_vertex
          //   = known cost from start to new_vertex
          // (determining the distance between curr_vertex and new_vertex using h
          // is only correct if they are directly adjacent because of possible obstacles)
          assert(EXPLORE_RADIUS <= 1);
          float new_g = __half2float(curr_vertex->f)
                  - h(curr_vertex->x, curr_vertex->y, goal.x, goal.y)
                  + h(curr_vertex->x, curr_vertex->y, new_x, new_y);

          half new_f = __float2half(new_g + h(new_x, new_y, goal.x, goal.y));

          /* printf("before: %.10f\n", __half2float(curr_vertex->f));
          printf("h(curr_vertex->x: %d, curr_vertex->y: %d, goal.x: %d, goal.y: %d): %.10f\n", curr_vertex->x, curr_vertex->y, goal.x, goal.y, h(curr_vertex->x, curr_vertex->y, goal.x, goal.y));
          printf("h(new_x: %d, new_y: %d, goal.x: %d, goal.y: %d): %.10f\n", new_x, new_y, goal.x, goal.y, h(new_x, new_y, goal.x, goal.y));
          printf("h(new_x: %d, new_y: %d, curr_vertex->x: %d, curr_vertex->y: %d)):  %.10f\n", new_x, new_y, curr_vertex->x, curr_vertex->y, h(new_x, new_y, curr_vertex->x, curr_vertex->y));
          printf("after: %.10f\n", __half2float(new_f));  */

          struct vertex new_vertex = {
            .x=new_x,
            .y=new_y,
            .f=new_f,
            .pred=curr_vertex,
            .node=agent
          };
          // printf("%d: looking at %d/%d, pred: %d/%d\n", threadIdx.x, new_x, new_y, curr_vertex->x, curr_vertex->y);

          ol_insert_list[tIdx] = new_vertex;
        }
      }
      __syncthreads();

      // deduplicate
      if(!tIdx)
      {

        for(int i = 0; i < NUM_THREADS_PER_AGENT; i++)
        {
          vertex *entry_a = &ol_insert_list[i];
          if(entry_a->x == -1)
            continue;
          for(int j = i + 1; j < NUM_THREADS_PER_AGENT; j++)
          {
            vertex *entry_b = &ol_insert_list[j];
            if(entry_a->x != entry_b->x || entry_a->y != entry_b->y)
              continue;

            // printf("%d/%d comes up twice: %.5f vs %.5f\n", entry_a->x, entry_a->y, __half2float(entry_a->f), __half2float(entry_b->f));
            if(__half2float(entry_a->f) > __half2float(entry_b->f))
            {
              entry_a->f = entry_b->f;
              entry_a->pred = entry_b->pred;
            }
            entry_b->x = -1;
          }
        }
      }

      __syncthreads();

      vertex entry = ol_insert_list[tIdx];

      // entry.x != -1 means a new vertex to insert into the ol was discovered
      if (entry.x != -1) {
        // printf("%d: checking for ins %d/%d %.2f\n", agent, entry.x, entry.y, entry.f);
        // only insert the newly-found vertex if it wasn't in the ol already or now has a better f
        bool better_f = __half2float(f_ol[agent][entry.y][entry.x]) == 0.0 || __half2float(f_ol[agent][entry.y][entry.x]) > __half2float(entry.f);
        __syncthreads();

        if (better_f) {
          // will update if the entry is already present, otherwise insert.
          // printf("%d: inserting %d/%d %.10f\n", agent, entry.x, entry.y, entry.f);

#if defined(A_STAR_REALTIME_VISUALIZATION) && !defined(A_STAR_REALTIME_VISUALIZATION_PATHS_ONLY)
        uchar4 pixel = { 255, (unsigned char)(sg_agent * 1234), (unsigned char)(sg_agent * 123), 255 };

        surf2Dwrite(pixel, surface, entry.x * sizeof(uchar4), entry.y);
#endif

#ifdef A_STAR_WRITE_IMAGES
        out_map[entry.y * A_STAR_GRID_X + entry.x] = sg_agent + 25;
#endif




          queue_insert_or_update(entry, agent);
          f_ol[agent][entry.y][entry.x] = entry.f;
        }
        __syncthreads();
      }
    }

    // is the execution of this agent completed?
    if (queue_is_empty(agent)) {
      if (threadIdx.x == 0) {
        atomicAdd(&completed_sg_agents, 1);
        if (found_goal[agent]) {
          atomicAdd(&sg_agents_with_path, 1);
        }

        if (/* VERBOSE_DEBUG && */ !found_goal[agent]) {
          // if(!agent)
            // printf("%d, %d: there is no path, closed list has size %d, head %d\n", agent, sg_agent, cl_length[agent], cl_head[agent]);
            // printf("%d, %d: there is no path\n", agent, sg_agent);
          queue_clear(agent);
        }
      }
    }
    __syncthreads();

#ifdef A_STAR_CALCULATE_QUEUE_LENGTH
    if(!threadIdx.x)
    {
      unsigned long long ql = queue_length(agent);
      atomicAdd(&queue_length_sum, ql);
      atomicAdd(&queue_length_observations, 1);
    }
    __syncthreads();
#endif

    if (VERBOSE_DEBUG) {
      if (threadIdx.x == 0) {
        printf("ol queue_length(%d) is %d\n", agent, queue_length(agent));
      }
      __syncthreads();
    }

  }
}


struct rgb {
  unsigned char r, g, b;
};

struct rgb pixel_to_rgb(int pixel)
{
  struct rgb col;
  col.r = pixel != 0 ? 255 : 0;
  col.g = (pixel * 1234) % 256;
  col.b = (pixel * 123) % 256;
  if(pixel == 1)
    col.r = col.g = col.b = 50;

  if (pixel == 2) {
    col.r = 255;
    col.g = col.b = 0;
  }

  return col;
}

void set_pixel(int x, int y, int *map, int pixel)
{
  if (x >= 0 && x < A_STAR_GRID_X && y >= 0 && y < A_STAR_GRID_Y) {
    map[y * A_STAR_GRID_X + x] = pixel;
  }
}

void dumpResult(int *map, struct vertex *h_goals)
{
  static int num = 0;

  typeof(sg_agents_with_path) h_sg_agents_with_path;
  cudaMemcpyFromSymbol(&h_sg_agents_with_path, sg_agents_with_path, sizeof(sg_agents_with_path));
  CudaCheckError();
  printf("Writing %d found paths to result file...\n", h_sg_agents_with_path);

  typeof(next_sg_agent) h_next_sg_agent;
  cudaMemcpyFromSymbol(&h_next_sg_agent, next_sg_agent, sizeof(next_sg_agent));
  CudaCheckError();

  const int maxStrLen = 1024;
  char header[maxStrLen];
  snprintf(header, maxStrLen, "P6\n%d %d 255\n", A_STAR_GRID_X, A_STAR_GRID_Y);

  for (int i = 0; i < h_next_sg_agent; i++) {
    int x = h_goals[i].x;
    int y = h_goals[i].y;

    int pixel = i + 5; // map[y * A_STAR_GRID_X + x];

    for(int r = 0; r < 7; r++)
    {
      for(int w = 0; w < 2; w++)
      {
        set_pixel(x - r + w, y - r, map, pixel);
        set_pixel(x - r + w, y + r, map, pixel);
        set_pixel(x + r + w, y - r, map, pixel);
        set_pixel(x + r + w, y + r, map, pixel);
      }
    }
  }

  char outfilename[maxStrLen];
  snprintf(outfilename, maxStrLen, "img/out%08d.pgm", num);
  num++;

  std::ofstream resultfile(outfilename, std::ofstream::binary);

  if (resultfile) {
    resultfile.write(header, strlen(header));

    if (!resultfile) {
      fprintf(stderr, "Error: header could be written to result file.\n");
      exit(-1);
    }

    for (int y = 0; y < A_STAR_GRID_Y; y++) {
      for (int x = 0; x < A_STAR_GRID_X; x++) {
        int pixel = map[y * A_STAR_GRID_X + x];
        struct rgb col = pixel_to_rgb(pixel);
        resultfile.write(reinterpret_cast<char*>(&col), 3 * sizeof(unsigned char));

        if (!resultfile) {
          fprintf(stderr, "Error: pixel (%d, %d) could not be written to result file.\n");
          exit(-1);
        }
      }
    }
  }
  else {
    fprintf(stderr, "Could not open result file.\n");
    // exit(-1);
  }
}



// -----------------------------
// Model interface
// -----------------------------

void model_init()
{
  cudaSetDevice(CUDA_DEVICE);
  CudaCheckError();

  printf("sizeof in_cl: %ld bytes, %d MiB\n", NUM_CONCURRENT_AGENTS * sizeof(cuda_bitset<vertex::MAX_ID>), NUM_CONCURRENT_AGENTS * sizeof(cuda_bitset<vertex::MAX_ID>) / 1024 / 1024);
  printf("sizeof f_ol: %ld bytes, %d MiB\n", sizeof(f_ol), sizeof(f_ol) / 1024 / 1024);

#ifdef A_STAR_TRACE_PATHS
  struct vertex *d_cl;
  CudaSafeCall(
      cudaMalloc(&d_cl, NUM_CONCURRENT_AGENTS * MAX_CL_BUFFER_SIZE * sizeof(struct vertex))
    );
  cudaMemcpyToSymbol(cl, &d_cl, sizeof(struct vertex *));
  printf("sizeof closed list: %ld bytes, %d MiB\n", NUM_CONCURRENT_AGENTS * MAX_CL_BUFFER_SIZE * sizeof(struct vertex), NUM_CONCURRENT_AGENTS * MAX_CL_BUFFER_SIZE * sizeof(struct vertex) / 1024 / 1024);
#endif

  /* __device__ struct vertex cl[NUM_CONCURRENT_AGENTS][MAX_CL_BUFFER_SIZE]; */

  cuda_bitset<vertex::MAX_ID> *d_in_cl;

  CudaSafeCall(
      cudaMalloc(&d_in_cl, NUM_CONCURRENT_AGENTS * sizeof(cuda_bitset<vertex::MAX_ID>))
    );

  cudaMemcpyToSymbol(in_cl, &d_in_cl, sizeof(cuda_bitset<vertex::MAX_ID> *));

  // load the map
  h_map = (int *)calloc(A_STAR_GRID_X * A_STAR_GRID_Y, sizeof(int));

  /*
  // load a specific map from file "a_star/map_small.pgm"
  char buf[A_STAR_GRID_X * A_STAR_GRID_Y];
  std::ifstream mapfile("a_star/map_small.pgm", std::ifstream::binary);
  if (mapfile) {
    mapfile.read(buf, A_STAR_GRID_X * A_STAR_GRID_Y);

    if (!mapfile) {
      fprintf(stderr, "Only %d bytes could be read from map file.\n", mapfile.gcount());
      exit(-1);
    }
  }
  else {
    fprintf(stderr, "Could not open map file.\n");
    exit(-1);
  }

  for (int i = 0; i < A_STAR_GRID_X * A_STAR_GRID_Y; i++) {
    h_map[i] = !buf[i];
  }
  */

  // randomly generate the map
  for(int i = 0; i < A_STAR_OBSTACLES; i++)
  {
    int x = rand() % A_STAR_GRID_X;
    int y = rand() % A_STAR_GRID_Y;
    int s = max(1, (int)(-1.0 * A_STAR_OBSTACLE_SIZE_MEAN * log((float)rand() / RAND_MAX)));

    for(int dx = 0; dx < s; dx++)
    {
      for(int dy = 0; dy < s; dy++)
      {
        if(x + dx >= A_STAR_GRID_X || y + dy >= A_STAR_GRID_Y)
          continue;
        h_map[(y + dy) * A_STAR_GRID_X + (x + dx)] = 1;
      }
    }
  }


  // find random start and goal coordinates for each agent which are not inside walls of the map
  int seed = 14580;
  printf("seed: %d\n", seed);
  srand(seed);

  h_starts = (struct vertex *)malloc(NUM_AGENTS * sizeof(struct vertex));
  h_goals  = (struct vertex *)malloc(NUM_AGENTS * sizeof(struct vertex));
  for (int i = 0; i < NUM_AGENTS; i++) {
    do {
      h_starts[i].x = rand() % A_STAR_GRID_X;
      h_starts[i].y = rand() % A_STAR_GRID_Y;
    } while (h_map[h_starts[i].y * A_STAR_GRID_X + h_starts[i].x]);
    memset(&h_starts[i].f, 0, 2);
    h_starts[i].pred = NULL;


    do {
      h_goals[i].x = rand() % A_STAR_GRID_X;
      h_goals[i].y = rand() % A_STAR_GRID_Y;
    } while (h_map[h_goals[i].y * A_STAR_GRID_X + h_goals[i].x]);
    // h_goals[i].f    = 0;
    memset(&h_goals[i].f, 0, 2);
    h_goals[i].pred = NULL;
  }
  CudaCheckError();

  // initialize device streams and copy data to device

  cudaMemcpyToSymbol(out_map, h_map, A_STAR_GRID_X * A_STAR_GRID_Y * sizeof(int));
  CudaCheckError();

  cudaMemcpyToSymbol(starts, h_starts, NUM_AGENTS * sizeof(struct vertex));
  CudaCheckError();

  cudaMemcpyToSymbol(goals, h_goals, NUM_AGENTS * sizeof(struct vertex));
  CudaCheckError();


  cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindSigned);

  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, A_STAR_GRID_X, A_STAR_GRID_Y);
  CudaCheckError();

  cudaMemcpyToArray(cuArray, 0, 0, h_map, A_STAR_GRID_X * A_STAR_GRID_Y * sizeof(int), cudaMemcpyHostToDevice);
  CudaCheckError();


  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeWrap;
  texDesc.addressMode[1]   = cudaAddressModeWrap;
  texDesc.filterMode       = cudaFilterModePoint;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  CudaCheckError();


#ifdef A_STAR_REALTIME_VISUALIZATION
  gl_interop_init();
  dim3 blocks(ceil((float)A_STAR_GRID_X / 256), A_STAR_GRID_Y);
  redraw_surface<<<blocks, 256>>>(texObj, bitmap_surface);
#endif

}


long model_get_events()
{
  typeof(completed_sg_agents) h_completed_sg_agents;
  cudaMemcpyFromSymbol(&h_completed_sg_agents, completed_sg_agents, sizeof(completed_sg_agents));
  CudaCheckError();
  return h_completed_sg_agents;
}


void model_handle_next()
{
#ifdef A_STAR_REALTIME_VISUALIZATION
  a_star<<<NUM_CONCURRENT_AGENTS, NUM_THREADS_PER_AGENT>>>(texObj, bitmap_surface);
#else
  a_star<<<NUM_CONCURRENT_AGENTS, NUM_THREADS_PER_AGENT>>>(texObj);
#endif

#ifdef A_STAR_REALTIME_VISUALIZATION
  cudaDeviceSynchronize();
  gl_interop_draw();
#endif

#ifdef A_STAR_WRITE_IMAGES
  static int iteration;
  if(iteration++ % A_STAR_WRITE_IMAGES_PERIOD == 0)
    model_finish();
#endif
}


void model_finish()
{
  cudaMemcpyFromSymbol(h_map, out_map, A_STAR_GRID_X * A_STAR_GRID_Y * sizeof(int));
  CudaCheckError();

  dumpResult(h_map, h_goals);

#ifdef A_STAR_CALCULATE_QUEUE_LENGTH
  unsigned long long h_queue_length_observations;
  unsigned long long h_queue_length_sum;
  cudaMemcpyFromSymbol(&h_queue_length_observations, queue_length_observations, sizeof(unsigned long long));
  CudaCheckError();
  cudaMemcpyFromSymbol(&h_queue_length_sum, queue_length_sum, sizeof(unsigned long long));
  CudaCheckError();

  printf("queue length observations: %ld, average length: %.2f\n", h_queue_length_observations, (double)h_queue_length_sum / h_queue_length_observations);
#endif
}




#endif // #ifdef _A_STAR
