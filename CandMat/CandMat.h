#ifndef __CANDMAT_H__
#define __CANDMAT_H__

#include <stdio.h>

class CandMat
{
public:
	CandMat();
	CandMat(int rows, int cols);
	CandMat(int rows, int cols, float val[]);

	// MATRIX
	static CandMat dotprod(CandMat m1, CandMat m2);
	static CandMat transpose(CandMat m);
	static CandMat scalar(float x, CandMat m);
	static CandMat add(CandMat m1, CandMat m2);
	static CandMat subtract(CandMat m1, CandMat m2);
	static CandMat map(float (*f)(float), CandMat m);
	static CandMat hadamardprod(CandMat m1, CandMat m2);

	void printMat();

	int rows;
	int cols;
	float val[40][40] = {};
	~CandMat();
};

#endif