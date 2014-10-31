#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <math.h>

#define MAXTHRDS 124
#define MAXMATRIXVALUE 100

typedef struct {
int my_start_row;
int my_end_row;
double **matrix_a;
double **matrix_b;
double **global_matrix;
int my_matrix_col_len;
} matrix_multip_t;


int main()
{
	double **A, **B, **C;
	int num_of_thrds, matrix_size, i,j,k,x;
	int submatrix_size;
	matrix_size = 128;
	time_t t;   
   	/* Intializes random number generator */
  	srand((unsigned) time(&t));

	num_of_thrds = omp_get_num_procs();
	omp_set_num_threads(num_of_thrds);

	
	submatrix_size = matrix_size/num_of_thrds;

	//allocate and initialize arrays
  	A =(double **)malloc(matrix_size*sizeof(double *));
  	A[0] = (double *)malloc(matrix_size*matrix_size*sizeof(double));

  	B =(double **)malloc(matrix_size*sizeof(double *));
  	B[0] = (double *)malloc(matrix_size*matrix_size*sizeof(double));

  	C =(double **)malloc(matrix_size*sizeof(double *));
  	C[0] = (double *)malloc(matrix_size*matrix_size*sizeof(double));

  	if (!A || !B || !C)
    {
      printf( "Out of memory, reduce matrix_size value.\n");
      exit(-1);
    }

    for(i=1; i<matrix_size; i++)
    {
      	A[i] = A[0]+i*matrix_size;
      	if (!A[i])
  		{
    		printf("memory failed \n");
    		exit(1);
  		}
    }
    for(i=1; i<matrix_size; i++)
    {
      	B[i] = B[0]+i*matrix_size;
      	if (!B[i])
  		{
    		printf("memory failed \n");
    		exit(1);
  		}
    }
    for(i=1; i<matrix_size; i++)
    {
      	C[i] = C[0]+i*matrix_size;
      	if (!C[i])
  		{
    		printf("memory failed \n");
    		exit(1);
  		}
    } 

  	// initialize the matrices
   	for(i=0; i<matrix_size; i++)
   	{
    	for(j=0; j<matrix_size; j++)
    	{
    		A[i][j] = i+j;
    		B[i][j] = i+j+10;
   		 	// we can also create random matrices using the rand() method from the math.h library
    		//A[i][j] = rand() % MAXMATRIXVALUE;
    		//B[i][j] = rand() % MAXMATRIXVALUE;

        	//printf("Value of A[%d][%d] %f\n",i,j,A[i][j]);
        	//printf("Value of B[%d][%d] %f\n",i,j,B[i][j]);
    	}
    }
	

	for(x=0; x<matrix_size; x++)
    {
    	cblas_dgemm(CblasRowMajor,
                CblasNoTrans, 
                CblasNoTrans, 
                1, matrix_size, matrix_size, 1.0,
                *A + x*matrix_size, matrix_size, 
                *B, matrix_size, 
                1.0, *C + x*matrix_size, 
                matrix_size);
    }


free(A);
free(B);
free(C);
}
