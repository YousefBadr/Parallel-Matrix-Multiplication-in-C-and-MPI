#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc , char * argv[])
{
    int my_rank;		/* rank of process	*/
    int p,i,j,row,col,k;			/* number of process	*/
    int portionSize1,rem1,arrSize ;
    int* subMat1 , *subMat2;
    int *mat1,*mat2,*mat2transpose,* subResMat;
    int choice;
    int mat1ROWS,mat1COl,mat2ROWS,mat2COl;
    double start_time,end_time;
    int Finalize=0;
    FILE *fptr;
    MPI_Status status;	/* return status for 	*/
    /* recieve		*/
    /* Start up MPI */
    MPI_Init( &argc , &argv );

    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of process */
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
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
        
        int x;
        x=mat1ROWS%p;
        if(x==0)
        {
            portionSize1=mat1ROWS * mat1COl/p;
            rem1 = mat1ROWS % p;
        }
        else
        {
            if(p>mat1ROWS)
                p=mat1ROWS;
            
            portionSize1=mat1ROWS * mat1COl/p;
            rem1 = mat1ROWS % p;
       
            if(portionSize1%mat1COl !=0)
            {
                for ( i = portionSize1; i > 0; i--)
                {
                    portionSize1--;
                    if(portionSize1%mat1COl==0)
                        break;
                }                
            }

        

        }
            
    }
    //reset p
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Bcast (&Finalize , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    
    if(Finalize==1)
    {
        MPI_Finalize();
        exit(1);
    }

    MPI_Bcast (&portionSize1 , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat1ROWS , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat1COl , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat2ROWS , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
    MPI_Bcast (&mat2COl , 1 , MPI_INT , 0 , MPI_COMM_WORLD);        

    mat1 = malloc (mat1ROWS *mat1COl *sizeof (int));
    mat2 = malloc (mat2ROWS *mat2COl *sizeof (int));

    if(my_rank==0)
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
    }

    subMat1 = malloc (p*portionSize1*sizeof (int));

    MPI_Scatter (mat1 ,portionSize1,MPI_INT,subMat1,portionSize1,MPI_INT,0,MPI_COMM_WORLD);

    subResMat = malloc (p*(portionSize1/mat1COl)*mat2COl*sizeof (int));
    
    mat2transpose = malloc (mat2ROWS *mat2COl*sizeof (int));
    if(my_rank==0)
    {
    for ( i = 0; i < mat2ROWS; i++)
        for ( j = 0; j < mat2COl; j++)
            mat2transpose[j*mat2ROWS+i] = mat2[i*mat2COl+j]; 
    }
    MPI_Bcast(mat2transpose,mat2ROWS * mat2COl, MPI_INT, 0, MPI_COMM_WORLD);

    k=0;
    i=0;
    int mm=0;
    while (mm<portionSize1/mat1COl)
    {
        int localsum=0;
        i=mm*mat1COl;
        for ( j = 0; j < mat2ROWS*mat2COl+1; j++)
        {
            if(j%mat2ROWS==0 && j!= 0)
            {
                i=mm*mat1COl;
                subResMat[k]=localsum;
                localsum=0;
                k++;
                if(j== mat2ROWS*mat2COl)break;
            }
            localsum+=(subMat1[i]*mat2transpose[j]);
            i++;
        }
        mm++;
    }

    int * globalResMat ;
    globalResMat = malloc (p*mat1ROWS*mat2COl*sizeof (int));
    
    //remainder part  
    if (my_rank == 0 && rem1 !=0)
    {
        mm=mat2ROWS-rem1+1;
        int Mat1StartIndex,resulrtmatrixstart;
        resulrtmatrixstart=(mat1ROWS*mat2COl)-(rem1*mat2COl);
        k=resulrtmatrixstart;
        while (mm<mat2ROWS+1)
        {
            int localsum=0;
            i=mm*mat1COl;
            for ( j = 0; j < mat2ROWS*mat2COl+1; j++)
            {
                if(j%mat2ROWS==0 && j!= 0)
                {
                    //printf("sum=%d, Mat1[%d]=%d, mat2transpose[%d]=%d,My_Rank= %d,k=%d \n",localsum,i,mat1[i],j,mat2transpose[j],my_rank,k);
                    i=mm*mat1COl;
                    globalResMat[k]=localsum;
                    localsum=0;
                    k++;
                    if(j== mat2ROWS*mat2COl)break;
                }
                localsum+=(mat1[i]*mat2transpose[j]);
                i++;
            }
            mm++;
        }        
    }

    MPI_Gather (subResMat ,(portionSize1/mat1COl)*mat2COl,MPI_INT, globalResMat,(portionSize1/mat1COl)*mat2COl,MPI_INT,0,MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        printf("\nResult Matrix is (%dX%d): \n",mat1ROWS,mat2COl);
        for (i = 0 ; i < mat1ROWS ; i++)
        {
            for (j = 0 ; j < mat2COl; j++)
            {
                printf("%d ",globalResMat[(i*mat2COl) + j]);
            }
            printf("\n");
        }
        end_time=MPI_Wtime();
        printf("\nRunning Time = %f\n\n", end_time-start_time);
    }
    if(my_rank==0 && choice ==1)
      fclose(fptr);

    /* shutdown MPI */
    MPI_Finalize();

    return 0;
}
