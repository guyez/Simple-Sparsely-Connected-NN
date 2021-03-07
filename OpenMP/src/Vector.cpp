/****************************************************************************
 *
 * Vector.cpp - AP4AI OpenMP project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#include "../include/Vector.hpp"
#include <cmath>

/*
 * Private members for reference
 *
 * unsigned int size_; // number of elements
 * float *vector;     // pointer to the array.
 *
 */

/**********************************************************
 * Constructors
 **********************************************************/

Vector::Vector(unsigned int n) : size_{n}
{
    vector_ = new float[size_]();
}

Vector::Vector(const Vector &rhs) : size_{rhs.size_}
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

Vector::Vector(Vector &&rhs) : size_{rhs.size_}, vector_{rhs.vector_}
{
    rhs.size_ = 0;
    rhs.vector_ = nullptr;
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
 * Other Functions
 **********************************************************/

unsigned int Vector::getNumOfElements() const { return size_; }

void Vector::printVector() const
{
    for (unsigned int i = 0; i < size_; i++)
    {

        std::cout << vector_[i] << "\t";
    }
    std::cout << "\n\n";
}
