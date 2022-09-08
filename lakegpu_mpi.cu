#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "mpi.h"

#define __DEBUG

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

int tpdt(double* t, double dt, double tf);

__device__ double f_gpu(double p, double t) { return -expf(-TSCALE * t) * p; }

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

__global__ void evolve_gpu(double* un, double* uc, double* uo, double* pebbles,
                           int* n_d, double* h_d, double *t_d, double *dt_d, int *rank_d, int *np_d) {
    int n = *n_d;
    int np = *np_d;
    int rank = *rank_d;
    double h = *h_d;

    int n_x = n / np;
    double t = *t_d;
    double dt = *dt_d;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int uc_idx = j + ((i+1) * n);
    int un_idx = j + (i * n);

    if ((j==0 && ((rank==0 && i!=0)||(rank == np-1 && i!=n_x-1))) ||
          (j==n-1 && ((rank == 0 && i!=0) || (rank == np-1 && i!=n_x-1)))) { 
        un[un_idx] = 0.;
    } else {
        un[un_idx] = 2*uc[uc_idx] - uo[un_idx] + VSQR *(dt * dt) *(( uc[uc_idx-1] + uc[uc_idx+1] + 
          uc[uc_idx + n] + uc[uc_idx - n] + 0.25*(uc[uc_idx - n - 1] + uc[uc_idx - n + 1] + uc[uc_idx + n + 1] + uc[uc_idx + n - 1])- 
          5 * uc[uc_idx])/(h * h) + f_gpu(pebbles[un_idx],t)); //TODO : rempve f_gpu and replace with f
    }
    uo[un_idx] = uc[uc_idx];
    uc[uc_idx] = un[un_idx];
}


void run_gpu(double *u, double *u0, double *u1, double *pebbles, int npoints, double h, double end_time, int nthreads, int rank, int np)
{
	cudaEvent_t kstart, kstop;
	float ktime;
        
	/* HW2: Define your local variables here */
  int *n_d, *rank_d, *np_d;
  double *h_d, *un_d, *uc_d, *uo_d, *pebbles_d;

  double t = 0.;
  double dt = h / 2.;
  double *t_d, *dt_d;
  
  int n_x = npoints / np;
  int n_y = npoints;

  double* ibuffer_recv = (double*)malloc(sizeof(double) * n_y);
  double* fbuffer_recv = (double*)malloc(sizeof(double) * n_y);
  double* ibuffer_send = (double*)malloc(sizeof(double) * n_y);
  double* fbuffer_send = (double*)malloc(sizeof(double) * n_y);
  
  memcpy(ibuffer_send, u1, sizeof(double) * n_y);
  memcpy(fbuffer_send, &u1[(n_x - 1) * n_y], sizeof(double) * n_y);

  /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
  cudaMalloc(&n_d, sizeof(int));
  cudaMalloc(&np_d, sizeof(int));
  cudaMalloc(&rank_d, sizeof(int));
  cudaMalloc(&h_d, sizeof(double));
  cudaMalloc(&t_d, sizeof(double));
  cudaMalloc(&dt_d, sizeof(double));

  cudaMalloc(&un_d, sizeof(double) * n_x * n_y);
  cudaMalloc(&uc_d, sizeof(double) * (n_x + 2) * n_y);
  cudaMalloc(&uo_d, sizeof(double) * n_x * n_y);
  cudaMalloc(&pebbles_d, sizeof(double) * n_x * n_y);


  cudaMemcpy(n_d, &npoints, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(np_d, &np, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(rank_d, &rank, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h_d, &h, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dt_d, &dt, sizeof(double), cudaMemcpyHostToDevice);
  
  cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n_x * n_y,cudaMemcpyHostToDevice);
  cudaMemcpy(uo_d, u0, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice);
  /* Leave first row in uc for ibuffer and final row for f buffer */
  cudaMemcpy(&uc_d[n_y], u1, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice);
  

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  dim3 blockShape = dim3(n_x / nthreads, npoints / nthreads);
  dim3 threadShape = dim3(nthreads, nthreads);
	/* HW2: Add main lake simulation loop here */
  MPI_Request reqs[4];
  MPI_Status statuses[4];
  

  while(1) {

    cudaMemcpy(t_d, &t, sizeof(double), cudaMemcpyHostToDevice);
        
    // if(rank > 0)
      MPI_Irecv(ibuffer_recv, n_y, MPI_DOUBLE, (rank - 1+np)%np , 0, MPI_COMM_WORLD, &reqs[0]);
    // if(rank < np-1)
      MPI_Isend(fbuffer_send, n_y, MPI_DOUBLE, (rank + 1)%np , 0,MPI_COMM_WORLD, &reqs[3]);
    

    // if(rank > 0)
      MPI_Wait(&reqs[0], &statuses[0]);
    // if(rank < np-1)
      MPI_Wait(&reqs[3], &statuses[3]);
    
    
    // if(rank < np-1)
      MPI_Irecv(fbuffer_recv, n_y, MPI_DOUBLE, (rank + 1)%np , 1, MPI_COMM_WORLD, &reqs[1]);
    // if(rank > 0)
      MPI_Isend(ibuffer_send, n_y, MPI_DOUBLE, (rank - 1+np)%np , 1,MPI_COMM_WORLD, &reqs[2]);
    
    
    // if(rank < np-1)
      MPI_Wait(&reqs[1], &statuses[1]);
    // if(rank > 0)
      MPI_Wait(&reqs[2], &statuses[2]);
    
    memset(ibuffer_send, 0.0, sizeof(double)*n_y);
    memset(fbuffer_send, 0.0, sizeof(double)*n_y);
    

  
    cudaMemcpy(uc_d, ibuffer_recv, sizeof(double) * n_y,cudaMemcpyHostToDevice);
    cudaMemcpy(&uc_d[(n_x+1) * n_y], fbuffer_recv, sizeof(double) * n_y,cudaMemcpyHostToDevice);

    evolve_gpu<<<blockShape, threadShape>>>(un_d, uc_d, uo_d, pebbles_d, n_d, h_d, t_d, dt_d, rank_d, np_d);

    // cudaMemcpy(uo_d, &uc_d[n_y], sizeof(double) * n_x * n_y, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(&uc_d[n_y], un_d, sizeof(double) * n_x * n_y,cudaMemcpyDeviceToDevice);
        
    cudaMemcpy(ibuffer_send, &uc_d[n_y], sizeof(double) * n_y, cudaMemcpyDeviceToHost);
    cudaMemcpy(fbuffer_send, &uc_d[n_x * n_y], sizeof(double) * n_y,cudaMemcpyDeviceToHost);

    memset(ibuffer_recv, 0.0, sizeof(double)*n_y);
    memset(fbuffer_recv, 0.0, sizeof(double)*n_y);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (!tpdt(&t, dt, end_time)) break;

  } 

  cudaMemcpy(&u[0], &un_d[0], sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost);
  
  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(n_d);
  cudaFree(h_d);
  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);
  cudaFree(pebbles_d);


	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
