#ifndef __FULLYCONNECTED_H__
#define __FULLYCONNECTED_H__

#include "Layer.h"

class fuco : public layer
{
public:
	fuco();
	fuco(int in, int out);
	CandMat forward(CandMat m1);
	CandMat solver(float lr, CandMat targ, int mode);
	CandMat solver(CandMat E, float lr, layer *lptr, int mode);

	void setWeight(CandMat m);
	void setBias(CandMat m);
	void setVWeight(CandMat m);
	void setVBias(CandMat m);
	void setInput(CandMat m);
	void setOutput(CandMat m);
	CandMat getWeight();
	CandMat getBias();
	CandMat getVWeight();
	CandMat getVBias();
	CandMat getInput();
	CandMat getOutput();
	~fuco();
};

#endif