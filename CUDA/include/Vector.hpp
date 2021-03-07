/****************************************************************************
 *
 * Vector.hpp - AP4AI CUDA project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#ifndef CUDA_VECTOR_HPP_
#define CUDA_VECTOR_HPP_
#include "../utils/hpc.h"
#include <iostream>
 
class Vector
{
public:
    /**********************************************************
     * Constructors
     **********************************************************/

    // Basic constructor to inisialize a Vector of size n.
    // All Vector positions will be initialized to 0.
    Vector(unsigned int n = 0);

    // COPY Vector
    Vector(const Vector &rhs);

    // Copy assignment operator
    Vector &operator=(const Vector &rhs);

    // MOVE Vector
    Vector(Vector &&rhs);

    // Move assignment operator
    Vector &operator=(Vector &&rhs);

    // dealloc vector_ 
    ~Vector();

    /**********************************************************
     * Operator Overloads
     **********************************************************/

    float &operator[](unsigned int n);

    const float &operator[](unsigned int n) const;

    friend bool operator==(Vector lhs, const Vector &rhs);
    
     /**********************************************************
     * Host and Device functions
     **********************************************************/
    // Move a Vector from the Device(CUDA GPU) to the Host
    void to_host();
    
    // Move a Vector from the Host to the Device(CUDA GPU)
    void to_cuda();

    /**********************************************************
     * Other Functions
     **********************************************************/

    // Get number of elements
    unsigned int getNumOfElements() const;

    // Get vector_ pointer
    float* getPointer() const;

    // Get current position
    bool on_cuda() const;

    // Print the vector to std::cout
    void printVector() const;

protected:
    unsigned int size_; 
    float *vector_;      
    bool on_cuda_;
};

#endif /* CUDA_VECTOR_HPP_ */
