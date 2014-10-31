#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#define MAXTHRDS 124


void malloc_matrix(double **m, int row, int column)
{
  int i;
  m =(double **)malloc(row*sizeof(double *));
  m[0] = (double *)malloc(row*column*sizeof(double));
  if(!m)
    {
      printf("memory failed \n");
      exit(1);
    }
  for(i=1; i<row; i++)
    {
      m[i] = m[0]+i*column;
      if (!m[i])
    {
    printf("memory failed \n");
    exit(1);
    }
    }
}

int main()
{

}