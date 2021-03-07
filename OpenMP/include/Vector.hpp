/****************************************************************************
 *
 * Vector.hpp - AP4AI OpenMP project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#ifndef VECTOR_HPP_
#define VECTOR_HPP_

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
     * Other Functions
     **********************************************************/

    // Get number of elements
    unsigned int getNumOfElements() const;

    // Print the vector to std::cout
    void printVector() const;

private:
    unsigned int size_;
    float *vector_;    
};

#endif /* VECTOR_HPP_ */
