/*
    doubler.c, Copyright (C) 2010 Stephen Makonin.
    
    Project: Doubler, side project for CMPT826

    A three-layer feed-forward back-propagation network 
    where the output number is twice the input number.
    
    Compiles w/o issue using GCC on Linux. Run:
        clear; gcc -lm -o Doubler Doubler.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/***** Global Settings and Definitions ****************************************/
#define SEQ_VECTOR_LEN 16
#define NOV_VECTOR_LEN 8

/***** Bit Manipulation Functions *********************************************/
#define BYTE_SIZE 8
#define MAX_BIT (BYTE_SIZE - 1)

#define get_bit(x, b) (float)((x >> (MAX_BIT - b)) & 1)
#define set_bit(x, b) (x | (1 << (MAX_BIT - b)))
#define clr_bit(x, b) (x & ~(1 << (MAX_BIT - b)))
#define put_bit(x, b, i) (i) ? set_bit(x, b) : clr_bit(x, b)

/***** Data Structures and Data Types *****************************************/
typedef unsigned char byte;

/***** Common Helper Functions ************************************************/
void print_array(const char *name, float *arr, int d1);
void print_2d_array(const char *name, float **arr, int d1, int d2);

/***** Back-Prop Net Data Structures ******************************************/
typedef struct
{
    int I, J, K;
    float r;
    float error_margin;
	
    float **Wij;
    float **Wjk;
    float **deltaWij;
    float **deltaWjk;
    float *Bj;
    float *Bk;
    float *Oi;
    float *Oj;
    float *Ok;
    float *d;
} bpnet_t;

/***** Back-Prop Net Functions ************************************************/
#define sigmoid(I) (1.0 / (1.0 + expf(-I)))
#define slope(O) (O * (1.0 - O))

int init_bpnet(bpnet_t *bpnet);
int free_bpnet(bpnet_t *bpnet);
int reset_bpnet(bpnet_t *bpnet);
float train_bpnet(bpnet_t *bpnet);
float test_bpnet(bpnet_t *bpnet);
int print_bpnet(bpnet_t *bpnet);

/***** Doubler Specific Globals ***********************************************/
#define SET_SIZE 25 //inital: 20
#define SPACIAL_SIZE (0x100 >> 1)
#define TEST_SIZE (SPACIAL_SIZE - SET_SIZE)

byte training_set[SET_SIZE][2] =
{
    {  2,   4},
    {  6,  12},
    { 40,  80},
    {  5,  10},
    {  3,   6},
    { 15,  30},
    {100, 200},
    { 50, 100},
    {  1,   2},
    { 75, 150},
    {  4,   8},
    {  8,  16},
    { 16,  32},
    { 32,  64},
    { 64, 128},
    { 70, 140},
    { 60, 120},
    { 80, 160},
    { 23,  46},//added
    {113, 226},//added
    { 94, 188},//added
    {110, 220},//added
    { 43,  86},//added
    { 45,  90},
    {  9,  18}
};

bpnet_t bpnet;
int rounds;
int auto_train_complete;
float training_accuracy;
float testing_accuracy;

void do_reset(void)
{
	reset_bpnet(&bpnet);

    training_accuracy = 0.0;
    testing_accuracy = 0.0;
    auto_train_complete = 0;
    rounds = 0;
}

void run_training(int num, int print)
{
	int i, j, k, set;
	byte I, O, D;
	float accuracy, a;

	while(num > 0)
	{
		accuracy = 0.0;		
		for(set = 0; set < SET_SIZE; set++)
		{
			//Load test set
			I = training_set[set][0];
			D = training_set[set][1];

            //Array-ify the input
            for(i = 0; i < bpnet.I; i++)
                bpnet.Oi[i] = (float)get_bit(I, i);
			
			//Array-ify desired output
			for(k = 0; k < bpnet.K; k++)
				bpnet.d[k] = (float)get_bit(D, k);
			
    		a = train_bpnet(&bpnet);
    		accuracy += a;
			
			//Encode the output
			for(k = 0; k < bpnet.K; k++)
				O = (bpnet.Ok[k] >= 0.5) ? set_bit(O, k) : clr_bit(O, k);

			if(num == 1 && print)
			{
				printf("Train Run: Input=%3d, Desired=%3d, Output=%3d ( ", I, D, O);
				for(k = 0; k < bpnet.K; k++)
					printf("%5.3f:%01d ", bpnet.Ok[k], (int)bpnet.d[k]);
				printf(") Error=%3d, Accuracy=%5.3f\n", abs(D - O), a);
			}
		}
		
		rounds++;
		accuracy = accuracy / (float)SET_SIZE * 100.0;

		if(num == 1 && print)
			printf("\nTrain accuracy = %f%% after %d training round(s).\n", accuracy, rounds);

		num--;
	}
	
	training_accuracy = accuracy;
}

void run_tests(int print, int errors_only)
{
	int i, j, k, set;
    byte I, O, D;
    float accuracy, a;

    accuracy = 0.0;
    for(set = 0; set < SPACIAL_SIZE; set++)
    {
        I = set;
        D = I * 2;

        j = 0;
        for(i = 0; i < SET_SIZE; i++)
        {
        	if(I == training_set[i][0])
		        j = 1;
        }

        if(j)
        	continue;
        	
		//Array-ify the input
        for(i = 0; i < bpnet.I; i++)
            bpnet.Oi[i] = (float)get_bit(I, i);
		
		//Array-ify desired output
		for(k = 0; k < bpnet.K; k++)
			bpnet.d[k] = (float)get_bit(D, k);
		
		a = train_bpnet(&bpnet);
		accuracy += a;
		
		//Encode the output
		for(k = 0; k < bpnet.K; k++)
			O = (bpnet.Ok[k] >= 0.5) ? set_bit(O, k) : clr_bit(O, k);	
		
		if(a >= 1.0)
		{
			if(errors_only)
				continue;
		}
		
		if(print)
		{
			printf("Test Run: Input=%3d, Desired=%3d, Output=%3d ( ", I, D, O);
			for(k = 0; k < bpnet.K; k++)
				printf("%5.3f:%01d ", bpnet.Ok[k], (int)bpnet.d[k]);
			printf(") Error=%3d, Accuracy=%5.3f\n", abs(D - O), a);
		}
	}

	accuracy = accuracy / (float)TEST_SIZE * 100.0;

	if(print)
		printf("\nTesting accuracy (letter avg) of size %d = %f%%.\n", TEST_SIZE, accuracy);
		
	testing_accuracy = accuracy;
}

float test_zero(void)
{
	int i, j, k;
    byte I, O, D;
    float accuracy;

    accuracy = 0.0;
    I = 0;
    D = I * 2;

	//Array-ify the input
    for(i = 0; i < bpnet.I; i++)
        bpnet.Oi[i] = (float)get_bit(I, i);
	
	//Array-ify desired output
	for(k = 0; k < bpnet.K; k++)
		bpnet.d[k] = (float)get_bit(D, k);
	
	accuracy = train_bpnet(&bpnet);
	
	return accuracy;
}

int main(void)
{
	int num;
	float a;

	bpnet.I = 8;
	bpnet.J = 16;
	bpnet.K = 8;
	bpnet.r = 0.5;
	bpnet.error_margin = 0.10;
	if(!init_bpnet(&bpnet))
		goto exit_program;
	printf("\nDoubler: I=%d J=%d k=%d\n", bpnet.I, bpnet.J, bpnet.K);

    training_accuracy = 0.0;
    testing_accuracy = 0.0;
    auto_train_complete = 0;
    rounds = 0;

	while(1)
	{
		printf("\n(1[0[0[0[0[0[0[0]]]]]]]) Train, (12) Auto Train, (2, 22) Run Test, (3) Debug, (4) Reset, (-1) Quit ? ");
		scanf("%d", &num);
		printf("\n");
		
		switch(num)
		{
			case -1:
				goto exit_program;
				break;

			case 1:
			case 10:
			case 100:
			case 1000:
			case 10000:
			case 100000:
			case 1000000:
			case 10000000:
				run_training(num, 1);
				break;

			case 2:
			case 22:
				run_tests(1, (num == 22));
				break;

			case 12:
				auto_train_complete = 0;
				while(auto_train_complete < 3 && rounds < 10000000)
				{
					run_training(1, 0);
					run_tests(0, 0);
					a = test_zero();
					
					printf("Round %7d, Accuracy: Training=%7.3f%%, Testing=%7.3f%%, Zero case=%5.3f\n", rounds, training_accuracy, testing_accuracy, a);

					if(training_accuracy >= 100.0 && testing_accuracy >= 100.0 && a >= 1.0)
						auto_train_complete++;
					else
						auto_train_complete = 0;
				}				
				break;

			case 3:
				print_bpnet(&bpnet);
				printf("Summary: %d Rounds, Accuracy: Training=%f%%, Testing=%f%%\n", rounds, training_accuracy, testing_accuracy);
				break;

			case 4:
				do_reset();
            	break;
		}
	}

exit_program:
	free_bpnet(&bpnet);
	return 1;
}

/***** Common Helper Functions ************************************************/
void print_array(const char *name, float *arr, int d1)
{
	int i;
	
	printf("%s: \t", name);
	for(i = 0; i < d1; i++)
		printf("%7.3f ", arr[i]);
	printf("\n");
}

void print_2d_array(const char *name, float **arr, int d1, int d2)
{
    int i, j;
	
    printf("%s:\n", name);
    for(i = 0; i < d1; i++)
    {
    	printf("\t");
    	for(j = 0; j < d2; j++)
    		printf("%7.3f ", arr[i][j]);
    	printf("\n");
    }
    printf("\n");
}

/***** Back-Prop Net Functions ************************************************/
int init_bpnet(bpnet_t *bpnet)
{	
	int i, j;
	
	if(!(bpnet->Wij = (float **)malloc(bpnet->I * sizeof(float *))))
		exit(0);
	
	for(i = 0; i < bpnet->I; i++)
        if(!(bpnet->Wij[i] = (float *)malloc(bpnet->J * sizeof(float *))))
        	exit(0);
	
	if(!(bpnet->Wjk = (float **)malloc(bpnet->J * sizeof(float *))))
		exit(0);
	
	for(j = 0; j < bpnet->J; j++)
    	if(!(bpnet->Wjk[j] = (float *)malloc(bpnet->K * sizeof(float *))))
    		exit(0);
    
	if(!(bpnet->deltaWij = (float **)malloc(bpnet->I * sizeof(float *))))
		exit(0);
	
	for(i = 0; i < bpnet->I; i++)
    	if(!(bpnet->deltaWij[i] = (float *)malloc(bpnet->J * sizeof(float *))))
    		exit(0);
	
	if(!(bpnet->deltaWjk = (float **)malloc(bpnet->J * sizeof(float *))))
		exit(0);
	
	for(j = 0; j < bpnet->J; j++)
    	if(!(bpnet->deltaWjk[j] = (float *)malloc(bpnet->K * sizeof(float *))))
    		exit(0);
	
	if(!(bpnet->Bj = (float *)malloc(bpnet->J * sizeof(float *))))
		exit(0);
	
	if(!(bpnet->Bk = (float *)malloc(bpnet->K * sizeof(float *))))
		exit(0);
	
	if(!(bpnet->Oi = (float *)malloc(bpnet->I * sizeof(float *))))
		exit(0);
	
	if(!(bpnet->Oj = (float *)malloc(bpnet->J * sizeof(float *))))
		exit(0);
	
	if(!(bpnet->Ok = (float *)malloc(bpnet->K * sizeof(float *))))
		exit(0);
	
	if(!(bpnet->d = (float *)malloc(bpnet->K * sizeof(float *))))
		exit(0);
	
	reset_bpnet(bpnet);
	
	return 1;
}

int reset_bpnet(bpnet_t *bpnet)
{
    int i, j, k;
	
    srand((unsigned)time(NULL));
	
    //Set random weight values
    for(j = 0; j < bpnet->J; j++)
    {
        for(i = 0; i < bpnet->I; i++)
            bpnet->Wij[i][j] = (float)((int)(rand() % 50 + 20)) / 100.0;
		
        for(k = 0; k < bpnet->K; k++)
            bpnet->Wjk[j][k] = (float)((int)(rand() % 50 + 20)) / 100.0;
    }
}

int free_bpnet(bpnet_t *bpnet)
{
	int i, j;
	
	for(i = 0; i < bpnet->I; i++)
		free(bpnet->Wij[i]);
	free(bpnet->Wij);
	
	for(j = 0; j < bpnet->J; j++)
		free(bpnet->Wjk[j]);
	free(bpnet->Wjk);
	
	for(i = 0; i < bpnet->I; i++)
		free(bpnet->deltaWij[i]);
	free(bpnet->deltaWij);
	
	for(j = 0; j < bpnet->J; j++)
		free(bpnet->deltaWjk[j]);
	free(bpnet->deltaWjk);
	
	free(bpnet->Bj);
	free(bpnet->Bk);
	free(bpnet->Oi);
	free(bpnet->Oj);
	free(bpnet->Ok);
	free(bpnet->d);
}

int query_bpnet(bpnet_t *bpnet)
{
    int i, j, k;
    float eta;
	
    //Fire activation for j level
    for(j = 0; j < bpnet->J; j++)
    {
        eta = 0;
        for(i = 0; i < bpnet->I; i++)
            eta += bpnet->Oi[i] * bpnet->Wij[i][j];
		
        bpnet->Oj[j] = sigmoid(eta);
    }
	
    //Fire activation for k level
    for(k = 0; k < bpnet->K; k++)
    {
        eta = 0;
        for(j = 0; j < bpnet->J; j++)
            eta += bpnet->Oj[j] * bpnet->Wjk[j][k];
		
        bpnet->Ok[k] = sigmoid(eta);        
    }
	
    return 1;
}

float compute_accuracy(bpnet_t *bpnet)
{
	int k;
	float ok, emin, emax, accuracy;
	
	ok = 0.0;
	for(k = 0; k < bpnet->K; k++)
	{
		emin = bpnet->d[k] - bpnet->error_margin;
		emax = bpnet->d[k] + bpnet->error_margin;
		
		if(emin < 0.0)
			emin = 0;
		
		if(emax > 1.0)
			emax = 1.0;
		
		if(bpnet->Ok[k] >= emin && bpnet->Ok[k] <= emax)
			ok += 1.0;
	}
	
	accuracy = ok / (float)bpnet->K;
	return accuracy;
}

float train_bpnet(bpnet_t *bpnet)
{
	int i, j, k;
	
	//Compute resulting output
	query_bpnet(bpnet);
	
	//Compute Beta for nodes in output layer
	for(k = 0; k < bpnet->K; k++)
		bpnet->Bk[k] = bpnet->d[k] - bpnet->Ok[k];
	
	//Compute Beta for all internal nodes
	for(j = 0; j < bpnet->J; j++)
	{
		bpnet->Bj[j] = 0.0;
		for(k = 0; k < bpnet->K; k++)
			bpnet->Bj[j] += bpnet->Wjk[j][k] * slope(bpnet->Ok[k]) * bpnet->Bk[k];
	}
	
	//Compute weight changes for all weights, i-j level
	for(i = 0; i < bpnet->I; i++)
		for(j = 0; j < bpnet->J; j++)
			bpnet->deltaWij[i][j] = bpnet->r * bpnet->Oi[i] * slope(bpnet->Oj[j]) * bpnet->Bj[j];
	
	//Compute weight changes for all weights, j-k level
	for(j = 0; j < bpnet->J; j++)
		for(k = 0; k < bpnet->K; k++)
			bpnet->deltaWjk[j][k] = bpnet->r * bpnet->Oj[j] * slope(bpnet->Ok[k]) * bpnet->Bk[k];
	
	//Add up the weight changes and change the weights
	for(j = 0; j < bpnet->J; j++)
	{
		for(i = 0; i < bpnet->I; i++)
			bpnet->Wij[i][j] += bpnet->deltaWij[i][j];
		
		for(k = 0; k < bpnet->K; k++)
			bpnet->Wjk[j][k] += bpnet->deltaWjk[j][k];
	}
	
	//Compute output accuracy
	return compute_accuracy(bpnet);
}

float test_bpnet(bpnet_t *bpnet)
{
	//Compute resulting output
	query_bpnet(bpnet);
	
	//Compute output accuracy
	return compute_accuracy(bpnet);
}

int print_bpnet(bpnet_t *bpnet)
{
	int i, j, k;
	
	printf("Debug print BP memory structures:\n\n");
	
	print_2d_array("Wij", bpnet->Wij, bpnet->I, bpnet->J);
	print_2d_array("Wjk", bpnet->Wjk, bpnet->J, bpnet->K);
	printf("\n");
	
	print_2d_array("deltaWij", bpnet->deltaWij, bpnet->I, bpnet->J);
	print_2d_array("deltaWjk", bpnet->deltaWjk, bpnet->J, bpnet->K);
	printf("\n");
	
	print_array("Bj", bpnet->Bj, bpnet->J);
	print_array("Bk", bpnet->Bk, bpnet->K);
	printf("\n");
	
	print_array("Oi", bpnet->Oi, bpnet->I);
	print_array("Oj", bpnet->Oj, bpnet->J);
	print_array("Ok", bpnet->Ok, bpnet->K);
	print_array("d", bpnet->d, bpnet->K);
	printf("\n");
	
	printf("Layer Nodes: I=%d, J=%d, K=%d\n", bpnet->I, bpnet->J, bpnet->K);
	printf("Learning Rate=%5.3f, Error Margin=%5.3f\n", bpnet->r, bpnet->error_margin);
	printf("\n");
}
