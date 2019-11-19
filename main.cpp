#include <iostream>
#include "CandNet.h"

// TRAINING MODE
#define SGD 			0
#define SGD_Momentum 		1

float input_[] =
{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
	.5, .5,
	.3, .3
};

float target_[] =
{
	0,
	1,
	1,
	0,
	.5,
	.3
};

float myIn[][2] = 
{
	{1, 1},			//0
	{0.7, 0.2},		//1
	{0, 0.1},		//2
	{0.9, 0.82},		//3
	{0, 1},			//4
	{0.9, .2}		//5
};

// INITIAL WEIGHT and BIAS
float weight_0[] = 
{
	19.3925914764, -18.9110946655, 
	19.0932750702, -19.6087875366 
};

float weight_1[] = 
{
	14.3811798096, -14.1888046265  
};

float bias_0[] = 
{
	-9.9959650040,
	10.0209493637
};

float bias_1[] = 
{
	6.9632215500
};
////////

int n_dataTrain = 6; //number of data training
int index = 0;

int main(int argc, char **argv)
{
	if(argc > 1)
	{
		index = atoi(argv[1]);
	}
	else
	{
		index = 0;
	}
	
	CandNet myNet(2, 2, 1); //CandNet myNet(number_input, number_hidden, number_output)
	CandMat myInput(2, 1, myIn[index]);

	//set weight and bias
	CandMat W0(2, 2, weight_0);
	CandMat W1(1, 2, weight_1);
	CandMat B0(2, 1, bias_0);
	CandMat B1(1, 1, bias_1);

	myNet.setAllWeight(W0, W1);
	myNet.setAllBias(B0, B1);

	// Data training
	// CandMat in(4, 2, input_); //matrix for input data train
	// CandMat targ(4, 1, target_); //matrix for target data train
	// printf("######### TRAINING..... ########\n");
	// myNet.training(n_dataTrain, 50000, in, targ, .9, SGD);
	// printf("######### RESULT ########\n");
	// (myNet.inference(myInput)).printMat();
	// myNet.training(n_dataTrain, 25000, in, targ, 1., SGD_Momentum);
	// printf("######### DONE TRAINING :) ########\n\n\n");

	printf("######### RESULT ########\n");
	(myNet.inference(myInput)).printMat();
	printf("\n");
	myNet.getAllWeight(); //print all the weight
	printf("\n");
	myNet.getAllBias(); //print all the bias
	return 0;
}
