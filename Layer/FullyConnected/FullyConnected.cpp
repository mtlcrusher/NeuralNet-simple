#include "FullyConnected.h"

fuco::fuco() : layer()
{

}

float initVal[] = {6, -2, 1, 2, -8, 3, 4, -1, 5, 3};

fuco::fuco(int in, int out) : layer(in, out)
{
	CandMat weight(out, in, initVal);
	CandMat bias(out, 1, initVal);
	CandMat vweight(out, in);
	CandMat vbias(out, 1);

	this->setWeight(weight);
	this->setBias(bias);

	this->setVWeight(vweight);
	this->setVBias(vbias);
}

CandMat fuco::forward(CandMat m1)
{
	this->setInput(m1);
	CandMat out = CandMat::map(sigmoid, (CandMat::add((CandMat::dotprod(this->getWeight(), m1)), this->getBias())));
	this->setOutput(out);
	return out;
}

CandMat fuco::solver(float lr, CandMat targ, int mode)
{
	CandMat error;
	error = CandMat::subtract(this->getOutput(), targ);
	// error = CandMat::subtract(targ, this->getOutput());

	switch(mode)
	{
		case 0:
			// WEIGHT
			this->setWeight(CandMat::subtract(
				this->getWeight(), CandMat::scalar(
					lr, 
					(CandMat::dotprod(
						CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput())), 
						CandMat::transpose(this->getInput()))))));

			// BIAS
			this->setBias(CandMat::subtract(
				this->getBias(), CandMat::scalar(
					lr, 
					(CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput()))))));
			break;

		case 1:

			// WEIGHT
			this->setVWeight(CandMat::add(
				CandMat::scalar(0.9, this->getVWeight()), 
				CandMat::scalar(
					0.1, 
					(CandMat::dotprod(
						CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput())), 
						CandMat::transpose(this->getInput()))))));

			// this->getVWeight().printMat();
			// getchar();

			this->setWeight(CandMat::subtract(
				this->getWeight(), 
				CandMat::scalar(
					lr,
					this->getVWeight())));

			// BIAS
			this->setVBias(CandMat::add(
				CandMat::scalar(0.9, this->getVBias()), 
				CandMat::scalar(
					0.1, 
					CandMat::hadamardprod(
						error, 
						CandMat::map(dsigmoid, this->getOutput())))));

			this->setBias(CandMat::subtract(
				this->getBias(), 
				CandMat::scalar(
					lr,
					this->getVBias())));
			break;
	}
	return error;
}

CandMat fuco::solver(CandMat E, float lr, int mode)
{
	CandMat error;
	error = CandMat::dotprod(CandMat::transpose(this->getWeight()), E);

	switch(mode)
	{
		case 0:
			// WEIGHT
			this->setWeight(CandMat::subtract(
				this->getWeight(), CandMat::scalar(
					lr, 
					(CandMat::dotprod(
						CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput())), 
						CandMat::transpose(this->getInput()))))));

			// BIAS
			this->setBias(CandMat::subtract(
				this->getBias(), CandMat::scalar(
					lr, 
					(CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput()))))));
			break;

		case 1:

			// WEIGHT
			this->setVWeight(CandMat::add(
				CandMat::scalar(0.9, this->getVWeight()), 
				CandMat::scalar(
					0.1, 
					(CandMat::dotprod(
						CandMat::hadamardprod(
							error, 
							CandMat::map(dsigmoid, this->getOutput())), 
						CandMat::transpose(this->getInput()))))));

			// this->getVWeight().printMat();
			// getchar();

			this->setWeight(CandMat::subtract(
				this->getWeight(), 
				CandMat::scalar(
					lr,
					this->getVWeight())));

			// BIAS
			this->setVBias(CandMat::add(
				CandMat::scalar(0.9, this->getVBias()), 
				CandMat::scalar(
					0.1, 
					CandMat::hadamardprod(
						error, 
						CandMat::map(dsigmoid, this->getOutput())))));

			this->setBias(CandMat::subtract(
				this->getBias(), 
				CandMat::scalar(
					lr,
					this->getVBias())));
			break;
	}
	return error;
}

void fuco::setWeight(CandMat m)
{
	this->weight = m;
}

void fuco::setBias(CandMat m)
{
	this->bias = m;
}

void fuco::setVWeight(CandMat m)
{
	this->vweight = m;
}

void fuco::setVBias(CandMat m)
{
	this->vbias = m;
}

void fuco::setInput(CandMat m)
{
	this->Input = m;
}

void fuco::setOutput(CandMat m)
{
	this->Output = m;
}

CandMat fuco::getWeight()
{
	return this->weight;
}

CandMat fuco::getBias()
{
	return this->bias;
}

CandMat fuco::getVWeight()
{
	return this->vweight;
}

CandMat fuco::getVBias()
{
	return this->vbias;
}

CandMat fuco::getInput()
{
	return this->Input;
}

CandMat fuco::getOutput()
{
	return this->Output;
}

fuco::~fuco()
{

}
