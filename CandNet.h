#ifndef __CANDNET_H__
#define __CANDNET_H__

#include "FullyConnected.h"

class CandNet
{
public:
	CandNet();
	CandNet(int in, int hidden, int out);
	CandMat inference(CandMat input);
	void training(int n_data, int epoch, CandMat input_, CandMat target_, float lr, int mode);
	void getAllWeight();
	void getAllBias();
	CandMat getWeight(int n);
	CandMat getBias(int n);
	void setAllWeight(CandMat W0, CandMat W1);
	void setAllBias(CandMat B0, CandMat B1);
	~CandNet();

protected:
	int in = 0;
	int hidden = 0;
	int out = 0;
	layer *lptr[2] = {};
};


#endif