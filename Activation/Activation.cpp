#include "Activation.h"

float sigmoid(float x)
{
	return 1/(1+exp(-x));
}

float dsigmoid(float y)
{
	return y*(1-y);
}

float ReLU(float x)
{
	if(x <= 0)
		return 0;
	else
		return x;
}

float dReLU(float x)
{
	return 0;
}

float linear(float m, float x)
{
	return m*x;
}