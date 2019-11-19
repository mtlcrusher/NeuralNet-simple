#include "Layer.h"

layer::layer()
{

}

layer::layer(int in, int out)
{
	this->input = in; 
	this->output = out;
}

layer::~layer()
{
	// cout << "layer destructor called" << endl;
}