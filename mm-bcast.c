#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc , char * argv[])
{
    int my_rank,NumberOfWorkers,processID;
    int p,i,j,k,startIndex;			
    int portionSize1,sentportionSize1,rem1;
    int *mat2transpose;
    int choice;
    int mat1ROWS,mat1COl,mat2ROWS,mat2COl;
    double start_time,end_time;
    int Finalize=0;
    FILE *fptr;
    MPI_Status status;	

    /* Start up MPI */
    MPI_Init( &argc , &argv );

    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of process */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    NumberOfWorkers = p-1;
    
    if(NumberOfWorkers==0)
    {
        if(my_rank==0)printf("\nMinimum number of processes is 2 because Process 0 works as a communicator,\nshuting down...\n");
        MPI_Finalize();
        exit(1);
    }
    if (my_rank == 0)
    {
        printf("\t\tWelcome to vector Matrix multiplication program!\n\n");
        printf("To read dimensions and values from file press 1\nTo read dimensions and values from console press 2\n>>");
        scanf("%d",&choice);
        if (choice == 2)
        {
            //first matrix
            printf("Please enter dimensions of the first matrix:");
            scanf("%d",&mat1ROWS);
            scanf("%d",&mat1COl);

            printf("\n");
            //second matrix
            printf("Please enter dimensions of the second matrix:");
            scanf("%d",&mat2ROWS);
            scanf("%d",&mat2COl);
        }
        else if (choice==1)
        {
            fptr=fopen("Data.txt","r");
            if(fptr == NULL)
            {
                printf("Error!");
                Finalize=1;
            }

            //first matrix
            fscanf(fptr,"%d",&mat1ROWS);
            fscanf(fptr,"%d",&mat1COl);

            //second matrix
            fscanf(fptr,"%d",&mat2ROWS);
            fscanf(fptr,"%d",&mat2COl);
            
        }
        if(mat1COl != mat2ROWS)
        {
            printf("\nMatrix 1 Number of Columns != Matrix 2 Number of Rows\n");
            Finalize=1;
        }
            
    }

    MPI_Bcast (&Finalize , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    
    if(Finalize==1)
    {
        MPI_Finalize();
        exit(1);
    }

    MPI_Bcast (&mat1ROWS , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat1COl , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat2ROWS , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat2COl , 1 , MPI_INT , 0 , MPI_COMM_WORLD);        

    int mat1 [mat1ROWS *mat1COl];
    int mat2 [mat2ROWS *mat2COl];
    int globalResMat [mat1ROWS *mat2COl];

    if (my_rank == 0)
   {          
        //Matrix 1
        if(choice==2)printf("Please enter Matrix 1 elements :\n");
        for (i = 0 ; i < mat1ROWS ; i++)
            for (j = 0 ; j < mat1COl; j++)
            {
                if(choice == 1)
                    fscanf(fptr,"%d",&mat1[(i*mat1COl) + j]);
                else
                    scanf("%d",&mat1[(i*mat1COl) + j]);
            }
                
        //Matrix 2
        if(choice==2)printf("Please enter Matrix 2 elements :\n");
        for (i = 0 ; i < mat2ROWS ; i++)
            for (j = 0 ; j < mat2COl; j++)
            {
                if(choice == 1)
                    fscanf(fptr,"%d",&mat2[(i*mat2COl) + j]);
                else
                    scanf("%d",&mat2[(i*mat2COl) + j]);
            }
        start_time = MPI_Wtime();   

      /* Send matrix data to the worker tasks */
      portionSize1 = mat1ROWS/NumberOfWorkers;
      rem1 = mat1ROWS%NumberOfWorkers;
      startIndex = 0;
      for (processID=1; processID<=NumberOfWorkers; processID++)
      {
         if(processID <= rem1)
            sentportionSize1=portionSize1+1;
         else
            sentportionSize1=portionSize1;
          	
         //printf("Sending %d rows to task %d startIndex=%d\n",sentportionSize1,processID,startIndex);
         MPI_Send(&startIndex, 1, MPI_INT, processID, 1, MPI_COMM_WORLD);
         MPI_Send(&sentportionSize1, 1, MPI_INT, processID, 1, MPI_COMM_WORLD);
         MPI_Send(&mat1[(startIndex*mat1COl)+0], sentportionSize1*mat1COl, MPI_INT, processID, 1,
                   MPI_COMM_WORLD);
         MPI_Send(&mat2, mat2ROWS*mat2COl, MPI_INT, processID, 1, MPI_COMM_WORLD);
         startIndex = startIndex + sentportionSize1;
      }

      /* Receive results from Slaves */
      for (i=1; i<=NumberOfWorkers; i++)
      {
         MPI_Recv(&startIndex, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
         MPI_Recv(&sentportionSize1, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
         MPI_Recv(&globalResMat[(startIndex*mat2COl)+0], sentportionSize1*mat2COl, MPI_INT, i, 2,MPI_COMM_WORLD, &status);
         //printf("Received results from task %d\n",i);
      }
      printf("Result Matrix:\n");
      for (i=0; i<mat1ROWS; i++)
      {
         printf("\n"); 
         for (j=0; j<mat2COl; j++) 
            printf("%d   ", globalResMat[(i*mat2COl) + j]);
      }
      end_time=MPI_Wtime();
      printf("\n\nRunning Time = %f\n\n", end_time-start_time);
   }
   
   /////////////////////////////////////////////////////Slaves////////////////////////////////////////////////////////////
   if (my_rank > 0)
   {
        MPI_Recv(&startIndex, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&sentportionSize1, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&mat1, sentportionSize1*mat1COl, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&mat2, mat2ROWS*mat2COl, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        for (k=0; k<mat2COl; k++)
            for (i=0; i<sentportionSize1; i++)
            {
                globalResMat[(i*mat2COl)+k] = 0;
                for (j=0; j<mat1COl; j++)
                globalResMat[(i*mat2COl)+k] = globalResMat[(i*mat2COl)+k] + mat1[(i*mat1COl)+j] * mat2[(j*mat2COl)+k];
            }
        MPI_Send(&startIndex, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&sentportionSize1, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&globalResMat, sentportionSize1*mat2COl, MPI_INT, 0, 2, MPI_COMM_WORLD);
   }


    if(my_rank==0 && choice ==1)
      fclose(fptr);

    /* shutdown MPI */
    MPI_Finalize();

    return 0;
}