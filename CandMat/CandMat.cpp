#include "CandMat.h"

// MATRIX
CandMat::CandMat()
{
	this->rows = 0;
	this->cols = 0;
}

CandMat::CandMat(int rows, int cols)
{
	this->rows = rows;
	this->cols = cols;
}
CandMat::CandMat(int rows, int cols, float val[])
{
	this->rows = rows;
	this->cols = cols;
	int k = 0;
	for(int i = 0; i < this->rows; i++)
	{
		for(int j = 0; j < this->cols; j++)
		{
			this->val[i][j] = val[k];
			k++;
		}
	}
}

CandMat CandMat::dotprod(CandMat m1, CandMat m2)
{
	CandMat res;
	for(int k = 0; k < m2.cols; k++)
	{
		for(int i = 0; i < m1.rows; i++)
		{
			for(int j = 0; j < m1.cols; j++)
			{
				res.val[i][k]+=(m1.val[i][j] * m2.val[j][k]);
			}
		}
	}

	res.rows = m1.rows;
	res.cols = m2.cols;
	return res;
}

CandMat CandMat::transpose(CandMat m)
{
	CandMat res;
	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
		{
			res.val[j][i] = m.val[i][j];
		}
	}
	res.rows = m.cols;
	res.cols = m.rows;
	return res;
}

CandMat CandMat::scalar(float x, CandMat m)
{
	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
		{
			m.val[i][j] = x*m.val[i][j];
		}
	}
	return m;
}

CandMat CandMat::add(CandMat m1, CandMat m2)
{
	for(int i = 0; i < m1.rows; i++)
	{
		for(int j = 0; j < m1.cols; j++)
		{
			m1.val[i][j] = m1.val[i][j] + m2.val[i][j];
		}
	}
	return m1;
}

CandMat CandMat::subtract(CandMat m1, CandMat m2)
{
	for(int i = 0; i < m1.rows; i++)
	{
		for(int j = 0; j < m1.cols; j++)
		{
			m1.val[i][j] = m1.val[i][j] - m2.val[i][j];
		}
	}
	return m1;
}

CandMat CandMat::map(float (*f)(float), CandMat m)
{
	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
		{
			m.val[i][j] = (*f)(m.val[i][j]);
		}
	}
	return m;
}

CandMat CandMat::hadamardprod(CandMat m1, CandMat m2)
{
	for(int i = 0; i < m1.rows; i++)
	{
		for(int j = 0; j < m1.cols; j++)
		{
			m1.val[i][j] = m1.val[i][j] * m2.val[i][j];
		}
	}
	return m1;
}

void CandMat::printMat()
{
	for(int i = 0; i < this->rows; i++)
	{
		for(int j = 0; j < this->cols; j++)
		{
			printf("%.10f ", this->val[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

CandMat::~CandMat()
{

}