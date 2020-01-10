#pragma once

#include <stdio.h>
#include "config.h"

#define MIN(a,b) (((a)<(b)) ? a : b)
#define MAX(a,b) (((a)>(b)) ? a : b)

// -----------------------------
// node -> LP mapping
// -----------------------------

static __host__ __device__ int get_lp(int node)
{
  return node / NUM_NODES_PER_LP;
}

// -----------------------------
// error checking
// -----------------------------

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

__host__ __device__
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
#ifdef __CUDA_ARCH__
        printf("cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
#else
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
#endif
    }
#endif

    return;
}

__host__ __device__
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
#ifdef __CUDA_ARCH__
        printf("cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
#else
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
#endif
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    /* err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
#ifdef __CUDA_ARCH__
        printf("cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
#else
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
#endif
    } */
#endif

    return;
}

#ifdef _TIMER
#define DECLARE_TIMER(s) struct timespec t1_##s, t2_##s; unsigned long ns_##s; double sum_##s; int count_##s;
#define START_TIMER(s) clock_gettime(CLOCK_MONOTONIC, &t1_##s);
#define STOP_TIMER(s) clock_gettime(CLOCK_MONOTONIC, &t2_##s); ns_##s = (t2_##s.tv_sec * 1e9 + t2_##s.tv_nsec) - (t1_##s.tv_sec * 1e9 + t1_##s.tv_nsec); sum_##s += ns_##s; count_##s++;

#define DECLARE_GPU_TIMER(s) cudaEvent_t t1_##s, t2_##s; float ms_##s; double sum_##s; int count_##s;
#define START_GPU_TIMER(s) cudaEventCreate(&t1_##s); cudaEventCreate(&t2_##s); cudaEventRecord(t1_##s);
#define STOP_GPU_TIMER(s) cudaEventRecord(t2_##s); cudaEventSynchronize(t2_##s); cudaEventElapsedTime(&ms_##s, t1_##s, t2_##s); sum_##s += ms_##s; count_##s++; cudaEventDestroy(t1_##s); cudaEventDestroy(t2_##s);
#define GET_SUM_MS_GPU(s) sum_##s

#define GET_AVG_TIMER(s) (double)(count_##s ? sum_##s/count_##s : 0)
#define GET_AVG_TIMER_MS(s) ((double)(count_##s ? sum_##s/count_##s : 0) / 1000000)
#define GET_AVG_TIMER_US(s) ((double)(count_##s ? sum_##s/count_##s : 0) / 1000)
#define GET_COUNT(s) count_##s
#define GET_SUM(s) sum_##s
#define GET_SUM_MS(s) (sum_##s / 1000000)
#define GET_TIMER(s) ns_##s
#define GET_TIMER_MS(s) (ns_##s / 1000000)
#define CLR_AVG_TIMER(s) sum_##s = 0; count_##s = 0

#else
#define DECLARE_TIMER(s) int count_##s;
#define START_TIMER(s)
#define STOP_TIMER(s) count_##s++;
#define GET_AVG_TIMER(s) 0.0
#define GET_AVG_TIMER_MS(s) 0.0
#define GET_AVG_TIMER_US(s) 0.0
#define GET_COUNT(s) count_##s
#define GET_SUM(s) 1.0
#define GET_SUM_MS(s) 1.0
#define GET_TIMER(s) 1.0
#define GET_TIMER_MS(s) 1.0
#define CLR_AVG_TIMER(s)
#endif

