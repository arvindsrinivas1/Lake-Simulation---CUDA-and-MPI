#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define ROOT 0 

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
double f(double p, double t);
void print_heatmap(const char *filename, double *u, int n, int rank, int np, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *un, double *uo, double *uc, double *pebbles, int npoints, double h, double end_time, int nthreads, int rank, int np);
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int rank, int np);

int main(int argc, char *argv[])
{

  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  int rank, np;
  /*Initialize MPI */
  MPI_Init(&argc, &argv);

  /*Get thenumber of procs in the comm*/
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  /*Get my rank in the comm */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);

  double *u_i0, *u_i1;
  double *u_cpu, *u_gpu, *pebs;
  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

  int n_x = npoints/np;
  int n_y = npoints;
  
  

  /* Storage fore results, each processor store it's own computed results. so, n_x * n_y) */

  u_cpu = (double*)malloc(sizeof(double) * npoints * n_y);
  u_gpu = (double*)malloc(sizeof(double) * n_x * n_y);

  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

  h = (XMAX - XMIN)/npoints;

  if(rank == ROOT)  {
    u_i0 = (double*)malloc(sizeof(double) * npoints * npoints);
    u_i1 = (double*)malloc(sizeof(double) * npoints * npoints);
    pebs = (double*)malloc(sizeof(double) * npoints * npoints);
    
    init_pebbles(pebs, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);

    for(int i=1; i<np; i++) {
      MPI_Send(&u_i0[i*n_x*n_y], n_x * n_y, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      MPI_Send(&u_i1[i*n_x*n_y], n_x * n_y, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      MPI_Send(&pebs[i*n_x*n_y], n_x * n_y, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
    }
    /*MPI Send pebbles, u_i0, u_i1) to other procs */
  } else {
    /*MPI Recv Pebbles u_i0, u_1, to other procs*/
    u_i0 = (double*)malloc(sizeof(double) * n_x * n_y);
    u_i1 = (double*)malloc(sizeof(double) * n_x * n_y);
    pebs = (double*)malloc(sizeof(double) * n_x * n_y);
    
    MPI_Status status;
    MPI_Recv(u_i0, n_x*n_y, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(u_i1, n_x*n_y, MPI_DOUBLE, ROOT, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(pebs, n_x*n_y, MPI_DOUBLE, ROOT, 2, MPI_COMM_WORLD, &status);
  }
  
  

  if(rank == ROOT)
    print_heatmap("lake_i.dat", u_i0, npoints, rank, 1, h);

  gettimeofday(&cpu_start, NULL);
  run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, rank, np);
  gettimeofday(&cpu_end, NULL);

  elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                  cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  printf("CPU took %f seconds\n", elapsed_cpu);
  char cfilename[255];
  sprintf(cfilename, "lake_f_cpu_%d.dat", rank);
  print_heatmap(cfilename, u_cpu, npoints, rank,  np, h);
  

  gettimeofday(&gpu_start, NULL);
  run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, rank, np);
  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  printf("GPU took %f seconds\n", elapsed_gpu);
  char filename[255];
  sprintf(filename, "lake_f_gpu_%d.dat", rank);
  print_heatmap(filename, u_gpu, npoints, rank,  np, h);

  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);
  free(u_gpu);

  MPI_Finalize();

  return 0;
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
                    uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void print_heatmap(const char *filename, double *u, int n, int rank, int np, double h)
{
  int i, j, idx;

  int n_x = n/np;
  int initialI = rank*n_x;

  FILE *fp = fopen(filename, "w");  

  for( i = 0; i < n_x; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", (initialI + i)*h, j*h, u[idx]);
    }
  }
  
  fclose(fp);
} 
