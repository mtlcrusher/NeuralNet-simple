#ifndef __LAYER_H__
#define __LAYER_H__

#include "CandMat.h"
#include "Activation.h"

class layer
{
public:
	layer();
	layer(int in, int out);
	virtual CandMat forward(CandMat m1) = 0;
	virtual CandMat solver(float lr, CandMat targ, int mode) = 0;
	virtual CandMat solver(CandMat E, float lr, int mode) = 0;

	virtual void setWeight(CandMat m) = 0;
	virtual void setBias(CandMat m) = 0;
	virtual void setVWeight(CandMat m) = 0;
	virtual void setVBias(CandMat m) = 0;
	virtual void setInput(CandMat m) = 0;
	virtual void setOutput(CandMat m) = 0;
	virtual CandMat getWeight() = 0;
	virtual CandMat getBias() = 0;
	virtual CandMat getVWeight() = 0;
	virtual CandMat getVBias() = 0;
	virtual CandMat getInput() = 0;
	virtual CandMat getOutput() = 0;
	virtual ~layer() = 0;

protected:
	int input;
	int output;
	CandMat weight;
	CandMat bias;
	CandMat Input;
	CandMat Output;
	CandMat vweight;
	CandMat vbias;
	
};

#endif