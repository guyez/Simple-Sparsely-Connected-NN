/****************************************************************************
 *
 * Vector.cu - AP4AI CUDA project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#include "../include/Vector.hpp"
#include <cmath>
#include <cuda.h>

/*
 * Private members for reference
 *
 * unsigned int size_; // number of elements
 * float *vector;      // pointer to the array
 * bool on_cuda_;      // boolean that used as a position tag
 *
 */

/**********************************************************
 * Constructors
 **********************************************************/

Vector::Vector(unsigned int n) : size_{n},  on_cuda_{false}
{
    vector_ = new float[size_]();
}

Vector::Vector(const Vector &rhs) : size_{rhs.size_}, on_cuda_{rhs.on_cuda_}
{
   vector_ = new float[size_]();
   std::copy(rhs.vector_, rhs.vector_ + size_, vector_);

}


Vector &Vector::operator=(const Vector &rhs)
{
    if (this != &rhs)
    {	
        Vector copy{rhs};
        std::swap(*this, copy);
    }
    return *this;
}

Vector::Vector(Vector &&rhs) : size_{rhs.size_}, vector_{rhs.vector_}, on_cuda_{rhs.on_cuda_}
{
    rhs.size_ = 0;
    rhs.vector_ = nullptr;
    rhs.on_cuda_ = false;
}

Vector &Vector::operator=(Vector &&rhs)
{
    if (this != &rhs) 
    {
        std::swap(size_, rhs.size_);
        std::swap(vector_, rhs.vector_);
    }
    return *this;
}

Vector::~Vector()
{
    if (on_cuda_)
        // Cleanup
        //cudaSafeCall(cudaFree(vector_));
        cudaFree(vector_);
    else
        delete[] vector_;
    
}

/**********************************************************
 * Operator Overloads
 **********************************************************/

float &Vector::operator[](unsigned int n)
{
    return vector_[n];
}

const float &Vector::operator[](unsigned int n) const
{
    return vector_[n];
}

/**********************************************************
 * Non-member, Friend Functions
 **********************************************************/

bool operator==(Vector lhs, const Vector &rhs)
{
    for (unsigned int i = 0; i < rhs.size_; i++)
    {
        if (!(std::fabs(lhs.vector_[i] - rhs.vector_[i]) < 0.000001))
        {
            return false;
        }
    }
    return true;
}

/**********************************************************
* Host and Device functions
**********************************************************/

void Vector::to_cuda() {
    float* h_vector = vector_;
    vector_ = nullptr;
    // Alloc space for device copy 
    //cudaSafeCall( cudaMalloc((void**) &vector_, size_ * sizeof(float)) );
    cudaMalloc((void**) &vector_, size_ * sizeof(float));
    // Copy to device
    //cudaSafeCall( cudaMemcpy(vector_, h_vector, size_ * sizeof(float), cudaMemcpyHostToDevice) );
    cudaMemcpy(vector_, h_vector, size_ * sizeof(float), cudaMemcpyHostToDevice);
    // Cleanup
    delete[] h_vector;
    on_cuda_ = true;
}


void Vector::to_host() {
    float* d_vector = vector_;
    vector_ = nullptr;
    // Alloc space for host copy 
    vector_ = new float[size_];
    // Copy to host
    //cudaSafeCall( cudaMemcpy(vector_, d_vector, size_ * sizeof(float), cudaMemcpyDeviceToHost) );
    cudaMemcpy(vector_, d_vector, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    // Cleanup
    //cudaSafeCall( cudaFree(d_vector) );
    cudaFree(d_vector);
    on_cuda_ = false;
}

/**********************************************************
 * Other Functions
 **********************************************************/

unsigned int Vector::getNumOfElements() const { return size_; }

float* Vector::getPointer() const { return vector_; }

bool Vector::on_cuda() const { return on_cuda_; }

void Vector::printVector() const
{
    for (unsigned int i = 0; i < size_; i++)
    {

    std::cout << vector_[i] << "\t";
    }
    std::cout << "\n\n";
}
