#include <stdlib.h>
#include <stdio.h>

void WriteData(char *fn, int n, double d)
{
   FILE *fp;
   fp = fopen(fn,"a");
   fprintf(fp, "%d,%lf\n", n, d);
   fclose(fp);
}


void deleteOutputFile(char *fn)
{
   int status;
   
   status = remove(fn);
 
   if( status == 0 )
      printf("%s file deleted successfully.\n",fn);
   else
   {
      printf("Unable to delete the file\n");
      perror("Error");
   }
}
