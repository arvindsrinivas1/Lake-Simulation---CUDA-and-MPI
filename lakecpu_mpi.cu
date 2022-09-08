#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"


#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

int tpdt(double* t, double dt, double tf);
double f(double p, double t);


void evolve9pt(double* un, double* uc, double* uo, double* pebbles,
                           int n, double h, double t, double dt, int rank, int np) {
    
    int n_x = n / np;
    int n_y = n;
    for(int i = 0; i < n_x; i++) {
        for(int j = 0; j < n_y; j++){
    
            int uc_idx = j + ((i+1) * n);
            int un_idx = j + (i * n);

            if ((j==0 && ((rank==0 && i!=0)||(rank == np-1 && i!=n_x-1))) ||
          (j==n-1 && ((rank == 0 && i!=0) || (rank == np-1 && i!=n_x-1)))) { 
                un[un_idx] = 0.;
            } else {
                un[un_idx] = 2*uc[uc_idx] - uo[un_idx] + VSQR *(dt * dt) *(( uc[uc_idx-1] + uc[uc_idx+1] + 
                uc[uc_idx + n] + uc[uc_idx - n] + 0.25*(uc[uc_idx - n - 1] + uc[uc_idx - n + 1] + uc[uc_idx + n + 1] + uc[uc_idx + n - 1])- 
                5 * uc[uc_idx])/(h * h) + f(pebbles[un_idx],t));
            }
        }
    }
}


void run_cpu(double *u, double *u0, double *u1, double *pebbles, int npoints, double h, double end_time, int nthreads, int rank, int np)
{
  double t = 0.;
  double dt = h / 2.;

  int n_x = npoints / np;
  int n_y = npoints;

  double *un, *uc, *uo;

  un = (double*)malloc(sizeof(double) * n_x * n_y);
  uc = (double*)malloc(sizeof(double) * (n_x+2) * n_y);
  uo = (double*)malloc(sizeof(double) * n_x * n_y);
  
  memcpy(uo, u0, sizeof(double) * n_x * n_y);
  memcpy(&uc[n_y], u1, sizeof(double) * n_x * n_y);

  double* ibuffer_recv = (double*)malloc(sizeof(double) * n_y);
  double* fbuffer_recv = (double*)malloc(sizeof(double) * n_y);
  double* ibuffer_send = (double*)malloc(sizeof(double) * n_y);
  double* fbuffer_send = (double*)malloc(sizeof(double) * n_y);
  
  memcpy(ibuffer_send, &uc[n_y], sizeof(double) * n_y);
  memcpy(fbuffer_send, &uc[(n_x - 1) * n_y], sizeof(double) * n_y);
  
  MPI_Request reqs[4];
  MPI_Status statuses[4];
  

  while(1) {

    // if(rank > 0)
      MPI_Irecv(ibuffer_recv, n_y, MPI_DOUBLE, (rank - 1+np)%np , 0, MPI_COMM_WORLD, &reqs[0]);
    // if(rank < np-1)
      MPI_Isend(fbuffer_send, n_y, MPI_DOUBLE, (rank + 1) %np, 0,MPI_COMM_WORLD, &reqs[3]);
    

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
    

    memcpy(uc, ibuffer_recv, sizeof(double) * n_y);
    memcpy(&uc[(n_x+1) * n_y], fbuffer_recv, sizeof(double) * n_y);
    
    evolve9pt(un, uc, uo, pebbles, npoints, h, t, dt, rank, np);

    memcpy(uo, &uc[n_y], sizeof(double) * n_x * n_y);
    memcpy(&uc[n_y], un, sizeof(double) * n_x * n_y);

    memcpy(ibuffer_send, &uc[n_y], sizeof(double) * n_y);
    memcpy(fbuffer_send,  &uc[n_x * n_y], sizeof(double) * n_y);

    memset(ibuffer_recv, 0.0, sizeof(double)*n_y);
    memset(fbuffer_recv, 0.0, sizeof(double)*n_y);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (!tpdt(&t, dt, end_time)) break;
  } 
  
  memcpy(&u[0], &un[0], sizeof(double) * n_x * n_y);
}
