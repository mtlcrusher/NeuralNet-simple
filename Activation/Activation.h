#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <math.h>

float sigmoid(float x);
float dsigmoid(float y);
float ReLU(float x);
float dReLU(float x);
float linear(float m, float x);

#endif