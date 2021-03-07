/****************************************************************************
 *
 * NeuralNet.hpp - AP4AI OpenMP project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#ifndef NEURALNET_HPP_
#define NEURALNET_HPP_

#include <random>
#include <omp.h>
#include "Vector.hpp"

#define R 5

class NeuralNet
{
public:
    // Default constructor to initialize a new neural net object
    NeuralNet(std::vector<unsigned int> topology, std::uniform_real_distribution<float> distr, std::mt19937 &generator);

    // Feed forward the input Vector and return the net's prediction
    Vector queryNet(const Vector &inputList, unsigned int p_mode);

private:
    // The activation function. Currently using Sigmoid function
    float activationFunction(float x) const;

    // Function used to randomly initialize the weight and bias vectors
    Vector initializeRandom(unsigned int n, std::uniform_real_distribution<float> distr, std::mt19937 &generator) const;

    Vector bias_;
    std::vector<unsigned int> topology_;
    std::vector<Vector> weights_;
    std::vector<Vector> outputs_;
};

#endif /* NEURALNET_HPP_ */
