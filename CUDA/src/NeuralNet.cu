/****************************************************************************
 *
 * NeuralNet.cu - AP4AI CUDA project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#include "../include/NeuralNet.hpp"

/*
 * Private members for reference
 * 
 * Vector bias_;
 * std::vector<unsigned int> topology_; 
 * std::vector<Vector> weights_;
 * std::vector<Vector> outputs_;
 *
 */

// The compiler will produces two versions of the function: one for the GPU and one for the CPU
__device__ __host__ float activationFunction(float x) {
    return (float) 1. / (1 + std::exp(-x));
}

// Kernel 
__global__ void forward(float* in, float* out,  unsigned int out_size, float* weights, float bias) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	float acc = 0;

	if (index < out_size)
	{
		// Compute the result
		for (unsigned int offset = 0; offset < R; offset++)
		{
			acc += in[index + offset] * weights[index * R + offset];
		}
		// Store the result 
		out[index] = activationFunction(acc + bias);
	}
}



// Kernel with shared memory
__global__ void forward_shared(float* in, unsigned int in_size, float* out,  unsigned int out_size,
								float* weights, float bias) {

	__shared__ float temp[BLKDIM + R - 1]; 
	
    unsigned int g_index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int l_index = threadIdx.x;
	
    float acc = 0;

    if (g_index < in_size) {
        // Read input elements into shared memory 
        temp[l_index] = in[g_index];
        if (g_index + BLKDIM < in_size && l_index < R - 1) 
            temp[l_index + BLKDIM] = in[g_index + BLKDIM];

        __syncthreads(); 

        if (g_index < out_size) {
			// Compute the result
            for (unsigned int offset = 0; offset < R; offset++) {
               acc += temp[l_index + offset] * weights[g_index * R + offset];
            }
			 // Store the result 
            out[g_index] = activationFunction(acc + bias);
        }
    }

}



/**********************************************************
 * Constructors
 **********************************************************/

NeuralNet::NeuralNet(std::vector<unsigned int> topology, std::uniform_real_distribution<float> distr, std::mt19937 &generator)
	: topology_{topology}, weights_{std::vector<Vector>()}, outputs_{std::vector<Vector>()}, on_cuda_{false}
{
	bias_ = initializeRandom(topology.size() - 1, distr, generator);

	for (unsigned int i = 0; i < topology.size(); i++)
	{
		if (i > 0)
		{
			weights_.push_back(initializeRandom(R * topology[i], distr, generator));
		}

		outputs_.push_back(Vector(topology[i]));
	}
}

/**********************************************************
* Host and Device functions
**********************************************************/

void NeuralNet::to_cuda() {

	for (unsigned int i = 0; i < topology_.size(); i++)
	{
		if (i > 0)
		{
			// Move all the weights and output layers to the GPU
			weights_[i-1].to_cuda();
			outputs_[i].to_cuda();
		}	
	}

    on_cuda_ = true;
}

void NeuralNet::to_host() {

	for (unsigned int i = 0; i < topology_.size(); i++)
	{
		// Move the weights and output layers to the host
		if (i > 0)
		{
			weights_[i-1].to_host();
		}
		if(outputs_[i].on_cuda() == true)
		{
			outputs_[i].to_host();
		}		
	}
    on_cuda_ = false;
}

/**********************************************************
 * Other Functions
 **********************************************************/


Vector NeuralNet::queryNet(const Vector &inputList, unsigned int s_mode) {

	if (on_cuda_)
	{
		// Copy the input in the first layer 
		outputs_[0] = inputList; 
		// Move the first layer to the GPU 
		outputs_[0].to_cuda();
		
		for (unsigned int i = 1; i < topology_.size(); i++)
		{
			// Kernel invocation
			if (s_mode) // Use shared memory 
			{
				forward_shared<<<(outputs_[i-1].getNumOfElements() + BLKDIM - 1) / BLKDIM, BLKDIM>>>(outputs_[i - 1].getPointer(),
					topology_[i - 1], outputs_[i].getPointer(),  topology_[i], weights_[i - 1].getPointer(), bias_[i - 1]);
			}
			else
			{
				forward<<<(outputs_[i-1].getNumOfElements() + BLKDIM - 1) / BLKDIM, BLKDIM>>>(outputs_[i - 1].getPointer(),
					outputs_[i].getPointer(), topology_[i], weights_[i - 1].getPointer(), bias_[i - 1]);
			}
			// Wait for kernel to finish 
			//cudaSafeCall(cudaDeviceSynchronize());  
			cudaDeviceSynchronize();
			
		}
	
		outputs_[0].to_host();
		outputs_.back().to_host();
	    return outputs_.back();

	} else {

		outputs_[0] = inputList;

		for (unsigned int i = 1; i < topology_.size(); i++)
		{
			float bias = bias_[i - 1];

				for (unsigned int j = 0; j < topology_[i]; j++)
				{
					float acc = 0;
					for (unsigned int offset = 0; offset < R; offset++)
					{
						acc += outputs_[i - 1][j + offset] * weights_[i - 1][j * R + offset];
					}
					outputs_[i][j] = activationFunction(acc + bias);
				}
		}
		return outputs_.back();	
	}

}

/**********************************************************
 * Private Functions
 **********************************************************/

Vector NeuralNet::initializeRandom(unsigned int n, std::uniform_real_distribution<float> distr, std::mt19937 &generator) const
{
	Vector init(n);

	for (unsigned int i = 0; i < n; i++)
	{
		init[i] = distr(generator);
	}
	return init;
}

