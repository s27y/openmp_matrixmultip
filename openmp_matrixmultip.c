#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <math.h>
#include <sys/time.h>

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


/* array memorry allocating
 * using omp static construct
 */
double** malloc_matrix(int matrix_size)
{
	int i;
  	double** m =(double **)malloc(matrix_size*sizeof(double *));
  	m[0] = (double *)malloc(matrix_size*matrix_size*sizeof(double));

  	if (!m)
    {
      printf( "Out of memory, reduce matrix_size value.\n");
      exit(-1);
    }

    #pragma omp parallel
    #pragma omp for schedule(static)
    for(i=1; i<matrix_size; i++)
    {
      	m[i] = m[0]+i*matrix_size;
      	if (!m[i])
  		{
    		printf("memory failed \n");
    		exit(1);
  		}
    }
    return m;
}

/* nitialize arrays
 * using omp dynamic construct
 */
void init_matrix(double** a, double** b, int matrix_size)
{
	int i,j;
	time_t t;
	/* Intializes random number generator */
  	srand((unsigned) time(&t));

  	#pragma omp parallel
  	#pragma omp for schedule(dynamic)
   	for(i=0; i<matrix_size; i++)
   	{
    	for(j=0; j<matrix_size; j++)
    	{
    		a[i][j] = i+j;
    		b[i][j] = i+j+10;
   		 	// we can also create random matrices using the rand() method from the math.h library
    		//A[i][j] = rand() % MAXMATRIXVALUE;
    		//B[i][j] = rand() % MAXMATRIXVALUE;

    		// comment out if need to check matrix value
        	//printf("Value of A[%d][%d] %f\n",i,j,a[i][j]);
        	//printf("Value of B[%d][%d] %f\n",i,j,b[i][j]);
    	}
    }
}

void multiply_matrix(double** a, double** b, double** c, int matrix_size)
{
	int i;
	for(i=0; i<matrix_size; i++)
    {
    	cblas_dgemm(CblasRowMajor,
                CblasNoTrans, 
                CblasNoTrans, 
                1, matrix_size, matrix_size, 1.0,
                *a + i*matrix_size, matrix_size, 
                *b, matrix_size, 
                1.0, *c + i*matrix_size, 
                matrix_size);
    }
}

void multiply_matrix_omp_dynamic(double** a, double** b, double** c, int matrix_size)
{
	int i;
	#pragma omp parallel
  	#pragma omp for schedule(dynamic)
	for(i=0; i<matrix_size; i++)
    {
    	cblas_dgemm(CblasRowMajor,
                CblasNoTrans, 
                CblasNoTrans, 
                1, matrix_size, matrix_size, 1.0,
                *a + i*matrix_size, matrix_size, 
                *b, matrix_size, 
                1.0, *c + i*matrix_size, 
                matrix_size);
    }
}

void multiply_matrix_omp_static(double** a, double** b, double** c, int matrix_size)
{
	int i;
	#pragma omp parallel
  	#pragma omp for schedule(static)
	for(i=0; i<matrix_size; i++)
    {
    	cblas_dgemm(CblasRowMajor,
                CblasNoTrans, 
                CblasNoTrans, 
                1, matrix_size, matrix_size, 1.0,
                *a + i*matrix_size, matrix_size, 
                *b, matrix_size, 
                1.0, *c + i*matrix_size, 
                matrix_size);
    }
}


int main()
{
	struct timeval tv1, tv2;
	struct timezone tz;
	double **A, **B, **C;
	int num_of_thrds, matrix_size, i,j,k,x;
	int submatrix_size;
	matrix_size = 128;

	num_of_thrds = omp_get_num_procs();
	omp_set_num_threads(num_of_thrds);

	// if we want each thread process submatrix_size of matrix
	// submatrix_size = matrix_size/num_of_thrds;

	A = malloc_matrix(matrix_size);
	B = malloc_matrix(matrix_size);
	C = malloc_matrix(matrix_size);

    init_matrix(A,B,matrix_size);	

    gettimeofday(&tv1, &tz);
    multiply_matrix_omp_dynamic(A,B,C,matrix_size);
    gettimeofday(&tv2, &tz);

    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Multiply takes %lf sec \n", elapsed);



    for(i=0; i<matrix_size; i++)
   	{
    	for(j=0; j<matrix_size; j++)
    	{
        	//printf("Value of C[%d][%d] %f\n",i,j,C[i][j]);
    	}
    }


free(A);
free(B);
free(C);
}
