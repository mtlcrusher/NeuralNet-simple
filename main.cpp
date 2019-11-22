#include <iostream>
#include "CandNet.h"

// MODE TRAINING
#define SGD 			0
#define SGD_Momentum 	1

// float input_[] =
// {
// 	0., 0.,
// 	0., 4090.,
// 	4090., 0.,
// 	4090., 4090.,
// 	2048., 2048.,
// 	1366., 1366.
// };

// float input_[] =
// {
// 	0, 0,
// 	0, 10,
// 	10, 0,
// 	10, 10,
// 	5, 5,
// 	3, 3
// };

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
	0.,
	1.,
	1.,
	0.,
	.5,
	.3
};

// float target_[] =
// {
// 	0.,
// 	4090.,
// 	4090.,
// 	0.,
// 	2048.,
// 	1366.
// };

float myIn[][2] = 
{
	{1, 1},			//0
	{.7, .2},		//1
	{0, .1},		//2
	{.8, .8},	//3
	{0, 1},			//4
	{.9, .2}		//5
};

// float myIn[][2] = 
// {
// 	{4090., 4090.},			//0
// 	{2866., 820.},		//1
// 	{0., 410.},		//2
// 	{3686., 3359.},	//3
// 	{0., 4090.},			//4
// 	{3686., 820.}		//5
// };

// float weight_0[] = 
// {
// 	19.3925914764, -18.9110946655, 
// 	19.0932750702, -19.6087875366 
// };

// float weight_1[] = 
// {
// 	14.3811798096, -14.1888046265  
// };

// float bias_0[] = 
// {
// 	-9.9959650040,
// 	10.0209493637
// };

// float bias_1[] = 
// {
// 	6.9632215500
// };

float weight_0[] = 
{
9.7055635452, -6.2900371552, 
2.1082117558, 3.1123290062, 
-14.7201337814, 18.0406112671, 
-22.8863430023, -16.1161899567
};

float weight_1[] = 
{
231.6437530518, -16.1603794098, 10.3650360107, 19.7140693665
};

float bias_0[] = 
{
-6.0194816589, 
13.4166574478, 
-9.7289333344, 
23.4331989288
};

float bias_1[] = 
{
-6.2314562798
};

const int n_dataTrain = 6;
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
	CandNet myNet(2, 4, 1);
	CandMat myInput(2, 1, myIn[index]);

	// //set weight and bias
	CandMat W0(4, 2, weight_0);
	CandMat W1(1, 4, weight_1);
	CandMat B0(4, 1, bias_0);
	CandMat B1(1, 1, bias_1);

	myNet.setAllWeight(W0, W1);
	myNet.setAllBias(B0, B1);

	// Data training
	CandMat in(n_dataTrain, 2, input_);
	CandMat targ(n_dataTrain, 1, target_);
	printf("######### SEDANG TRAINING..... ########\n");
	// myNet.training(n_dataTrain, 500000, in, targ, .8, SGD);
	// printf("######### HASIL ########\n");
	// (myNet.inference(myInput)).printMat();
	myNet.training(n_dataTrain, 2, in, targ, 1., SGD_Momentum);
	// printf("######### TRAINING SELESAI :) ########\n\n\n");

	printf("######### HASIL ########\n");
	(myNet.inference(myInput)).printMat();
	printf("\n");
	myNet.getAllWeight();
	printf("\n");
	myNet.getAllBias();
	return 0;
}