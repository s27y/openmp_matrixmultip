#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <math.h>
#include <sys/time.h>
#include "io.c"

#define MAXTHRDS 124
#define MAXMATRIXVALUE 100
#define OMP_STATIC_OUTPUT_FILENAME "out_omp_static.csv"
#define OMP_DYNAMIC_OUTPUT_FILENAME "out_omp_dynamic.csv"
#define SERIAL_OUTPUT_FILENAME "out_serial.csv"
#define RESULT_CHECK_FILENAME "out_result"
#define LOOP_START 4
#define LOOP_END 11
#define OMP_DYNAMIC "dynamic"
#define OMP_STATIC "static"
#define SERIAL "serial"


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
 *
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

/* initialize arrays
 * using omp dynamic construct
 *
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

/* multiply matrix using blas - serial
 *
 */
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

/* multiply matrix using blas - dynamic
 *
 */
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

/* multiply matrix using blas - static
 *
 */
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

/* calculate matrix one norm - omp
 *
 */
int matrix_one_norm_omp(double** m,int matrix_size)
{
	int i,j;

	int max = 0;
	int col_sum;
	#pragma omp parallel for
	#pragma omp shared(max)
	#pragma omp private(col_sum)
	for (i = 0; i < matrix_size; i++)
	{
		col_sum = 0;
		#pragma omp parallel for reduction(+: col_sum)
		for (j = 0; j < matrix_size; j++)
		{
			col_sum += abs(m[j][i]);
		}
		//printf("col_sum: %d\n", col_sum);
		if (col_sum > max)
		{
			{
				#pragma omp critical
				max = col_sum;
			}
			
		}
	}
	return max;
}

/* calculate matrix one norm - serial version
 *
 */
int matrix_one_norm(double** m,int matrix_size)
{
	int i,j;

	int max = 0;

	for (i = 0; i < matrix_size; i++)
	{
		int col_sum = 0;
		for (j = 0; j < matrix_size; j++)
		{
			col_sum += abs(m[j][i]);
		}
		//printf("col_sum: %d\n", col_sum);

		if (col_sum > max)
		{
			max = col_sum;
		}
	}
	return max;
}


int main(int argc, char *argv[])
{
	struct timeval tv1, tv2;
	struct timezone tz;
	double **A, **B, **C;
	int numreps,num_of_thrds, matrix_size, i,j,k,x;
    int one_norm_omp_dynamic, one_norm_omp_static, one_norm_serial;
	int submatrix_size;
    double average, elapsed;
	matrix_size = 1024;

	num_of_thrds = omp_get_num_procs();
	omp_set_num_threads(num_of_thrds);
	printf("Number of thread: %d\n", num_of_thrds);
    numreps = 3;

	// if we want each thread process submatrix_size of matrix
	// submatrix_size = matrix_size/num_of_thrds;


    if( argc == 2 )
   {
      printf("The argument supplied is '%s'\n", argv[1]);
      if(strcmp(argv[1],OMP_DYNAMIC) != 0 || strcmp(argv[1],OMP_STATIC) != 0 || strcmp(argv[1],SERIAL) != 0)
      {
        printf("The argument supplied is not  %s, %s or %s \n", OMP_DYNAMIC, OMP_STATIC, SERIAL);
      }
   }
   else if( argc > 2 )
   {
      printf("Too many arguments supplied.\n");
   }
   else
   {
      printf("One argument expected.\n");
   }


    if(strcmp(argv[1],OMP_DYNAMIC) == 0)
    {

    deleteOutputFile(OMP_STATIC_OUTPUT_FILENAME);
    for(i = LOOP_START; i < LOOP_END; i++)
    {
        matrix_size = pow(2, i);

        A = malloc_matrix(matrix_size);
        B = malloc_matrix(matrix_size);
        C = malloc_matrix(matrix_size);
        init_matrix(A,B,matrix_size);

        for(j = 0; j < numreps; j++)
        {
            // omp dynamic
            gettimeofday(&tv1, &tz);
            multiply_matrix_omp_static(A,B,C,matrix_size);
            one_norm_omp_static = matrix_one_norm_omp(C, matrix_size);
            gettimeofday(&tv2, &tz);
            elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
            average += elapsed;
            printf("matrix size %d, matrix_one_norm_omp takes %lf sec \n", matrix_size, elapsed);

        }
        average= average/3;

        WriteData(OMP_STATIC_OUTPUT_FILENAME, matrix_size, average);
       
    }
    WriteData(RESULT_CHECK_FILENAME, 1, one_norm_omp_static);

    free(A);
    free(B);
    free(C);
    elapsed = 0;
    average =0;

    }

    if(strcmp(argv[1],OMP_STATIC) == 0)
    {
    deleteOutputFile(OMP_DYNAMIC_OUTPUT_FILENAME);
    for(i = LOOP_START; i < LOOP_END; i++)
    {
        matrix_size = pow(2, i);

        A = malloc_matrix(matrix_size);
        B = malloc_matrix(matrix_size);
        C = malloc_matrix(matrix_size);
        init_matrix(A,B,matrix_size);

        for(j = 0; j < numreps; j++)
        {
            // omp dynamic
            gettimeofday(&tv1, &tz);
            multiply_matrix_omp_dynamic(A,B,C,matrix_size);
            one_norm_omp_dynamic = matrix_one_norm_omp(C, matrix_size);
            gettimeofday(&tv2, &tz);
            
            elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
            average += elapsed;
            printf("matrix size %d, matrix_one_norm_omp takes %lf sec \n", matrix_size, elapsed);
        }
        average= average/3;
        WriteData(OMP_DYNAMIC_OUTPUT_FILENAME, matrix_size, average);
    }
    WriteData(RESULT_CHECK_FILENAME, 2, one_norm_omp_dynamic);
    free(A);
    free(B);
    free(C);
    elapsed = 0;
    average = 0;
    }

    if(strcmp(argv[1],SERIAL) == 0)
    {
    deleteOutputFile(SERIAL_OUTPUT_FILENAME);
    for(i = LOOP_START; i < LOOP_END; i++)
    {
        matrix_size = pow(2, i);

        A = malloc_matrix(matrix_size);
        B = malloc_matrix(matrix_size);
        C = malloc_matrix(matrix_size);
        init_matrix(A,B,matrix_size);

        for(j = 0; j < numreps; j++)
        {
        // omp dynamic
        gettimeofday(&tv1, &tz);
        multiply_matrix(A,B,C,matrix_size);
        one_norm_serial = matrix_one_norm(C, matrix_size);
        gettimeofday(&tv2, &tz);

        elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
        average += elapsed;
        printf("matrix size %d, matrix_one_norm_omp static takes %lf sec \n", matrix_size, elapsed);

        }
        average = average/3;
        WriteData(SERIAL_OUTPUT_FILENAME, matrix_size, average);
    }
    WriteData(RESULT_CHECK_FILENAME, 3, one_norm_serial);
    free(A);
    free(B);
    free(C);
    }

}
