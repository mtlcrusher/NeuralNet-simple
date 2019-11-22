#include "CandNet.h"
#include <iostream>

CandNet::CandNet()
{

}

CandNet::CandNet(int in, int hidden, int out)
{
	this->in = in;
	this->hidden = hidden;
	this->out = out;

	int i = 0;
	for(i = 0; i < 1; i++)
	{
		out = hidden;
		lptr[i] = new fuco(in, out);
		in = hidden;
	}
	out = this->out;
	lptr[i] = new fuco(in, out);
}

CandMat CandNet::inference(CandMat input)
{
	return lptr[1]->forward(lptr[0]->forward(input));
}

void CandNet::training(int n_data, int epoch, CandMat input_, CandMat target_, float lr, int mode)
{
	for(int ep = 0; ep < epoch; ep++)
	{
		for(int data = 0; data < n_data; data++)
		{
			CandMat INPUT(this->in, 1, input_.val[data]);
			CandMat TARGET(this->out, 1, target_.val[data]);
			inference(INPUT);
			lptr[0]->solver(lptr[1]->solver(lr, TARGET, mode), lr, lptr[1], mode);
		}
	}
}

void CandNet::getAllWeight()
{
	printf("######## WEIGHT_0 #######\n");
	(lptr[0]->getWeight()).printMat();
	printf("\n");
	printf("######## WEIGHT_1 #######\n");
	(lptr[1]->getWeight()).printMat();
	printf("\n");
}

void CandNet::getAllBias()
{
	printf("######## BIAS_0 #######\n");
	(lptr[0]->getBias()).printMat();
	printf("\n");
	printf("######## BIAS_1 #######\n");
	(lptr[1]->getBias()).printMat();
	printf("\n");
}

void CandNet::setAllWeight(CandMat W0, CandMat W1)
{
	lptr[0]->setWeight(W0);
	lptr[1]->setWeight(W1);
}

void CandNet::setAllBias(CandMat B0, CandMat B1)
{
	lptr[0]->setBias(B0);
	lptr[1]->setBias(B1);
}

CandMat CandNet::getWeight(int n)
{
	return lptr[n]->getWeight();
}

CandMat CandNet::getBias(int n)
{
	return lptr[n]->getBias();
}


CandNet::~CandNet()
{

}