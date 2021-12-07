// -*- lsst-c++ -*-
#define WRAP_PYTHON 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#if WRAP_PYTHON
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#endif
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "TSPNode.h"

using namespace std;
using namespace Eigen;
typedef Array<bool,Dynamic,1> ArrayXb;
/**
 * @brief Implementation of an histogram-based mutual information estimator using data-driven tree-structured partitions (TSP).
 * 
 * @see [Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation](https://sail.usc.edu/publications/files/silva_tit_2012.pdf).
 */
class TSP {
    private:
        // Exponent of the tree-structured partitions (TSP) refinement criterion threshold approximation @f$b_n \approx n^{-l}@f$.
        double l_bn;
        // Weighting factor of the tree-structured partitions (TSP) refinement criterion threshold approximation @f$b_n = w * n^{-l}@f$.
        double w_bn;
        // Tree-structured partitions (TSP) regularization factor.
        double lambda;
        // Root node pointer of the tree-structured partitions (TSP).
        TSPNode *root;
        // Dataset dimensionality.
        unsigned int dim;
        // Dataset number of samples.
        unsigned int n_samples;
        // Tree-structured partitions (TSP) number of leaves.
        unsigned int tsp_size;
        // Tree-structured partitions (TSP) estimated mutual information.
        double tsp_emi;
        // Tree-structured partitions (TSP) regularized mutual information estimation.
        double tsp_reg_emi;
        /**
        * @brief Calculates the estimated mutual information from the minimum cost trees of sizes k={1,...,|T_full|}, where each optimal tree of size k is obtained by the split of the leave of maximum conditional information gain of the tree of size k-1 according to the subbaditive penalty property.
        * 
        * @param internalNodesArray Ordered array of the lowest leaf of the minimum cost trees of sizes k={1,...,|T_full|}.
        * @param optimalEMIArray Ordered array of the estimated mutual information from the minimum cost trees of sizes k={1,...,|T_full|}.
        * @return Nothing.
        * @see Equation (47) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
        */
        void minimumCostTrees(TSPNode **internalNodesArray, double *optimalEMIArray);
        /**
         * @brief Insert a node into the array according to an increasing order of the node's conditional mutual information gain.
         * 
         * @param node Node to insert.
         * @param internalNodesArray Array of nodes.
         * @param size Number of nodes from array.
         * @return Nothing.
         */
        void subadditiveInsert(TSPNode* node, TSPNode** internalNodesArray, unsigned int size);
    public:
        /**
         * @brief Class constructor.
         * 
         * @param l_bn Exponent of the TSP cells refinement criterion threshold approximation @f$b_n \approx n^{-l}@f$.
         * @param w_bn Weighting factor of the TSP cells refinement criterion threshold approximation @f$b_n = w * n^{-l}@f$.
         * @param lambda Tree-structured partitions (TSP) regularization factor.
         * @throw Invalid argument exception if parameter `l` do not belong to the interval (0, 1/3).
         */
        TSP(double l_bn, double w_bn, double lambda);
        /**
         * @brief Class destructor.
         */
        ~TSP();
        /**
         * @brief Generates data-driven tree-structured partitions (TSP) for mutual information estimation between two random variables samples `X` and `Y` according to the growth stopping criterion.
         * 
         * @param X First random variable sample set.
         * @param Y Second random variable sample set.
         * @return Nothing.
         * @see Equation (8) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        void grow(MatrixXd X, MatrixXd Y);
        /**
         * @brief Complexity-regularization of the tree-structured partition (TSP).
         * 
         * @return Nothing.
         * @see Equation (43) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        void regularize();
        /**
         * @brief Histogram-based mutual information estimation of the two random variables samples I(X,Y) using the data-driven tree-structured partitions (TSP).
         * 
         * @return Estimated mutual information.
         * @throw Invalid argument exception if the tree-structured partitions (TSP) has not been generated.
         * @see Equation (5) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        double emi();
        /**
         * @brief Calculates the regularized expression of the optimal mutual information estimation.
         *
         * @return Regularized mutual information estimation.
         * @throw Invalid argument exception if the tree-structured partitions (TSP) has not been generated.
         * @see Equation (32) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        double reg_emi();
        /**
         * @brief Calculates the number of leaves of the tree-structured partitions (TSP).
         * 
         * @return Number of leaves.
         * @throw Invalid argument exception if the tree-structured partitions (TSP) has not been generated.
         */
        unsigned int size();
        /**
         * @brief **Prints** two matrices with the lower and upper bounds of each partition of the full TSP.
         * 
         * @return Nothing.
         */
        void partitionsBounds();

        #if WRAP_PYTHON
        /**
         * @brief Python to C++ grow function wrapper. Converts the data Python objects into Eigen objects and then runs the `grow` class method.
         * 
         * @param X First random variable sample set.
         * @param Y Second random variable sample set.
         * @return Nothing.
         */
	    void grow_python(PyObject* X, PyObject* Y);
        #endif
};

#if WRAP_PYTHON
void TSP::grow_python(PyObject* X, PyObject* Y) {
    /* TODO: Concatenate PyArrayObjects to operate with PyArrayObjects instead of Eigen::MatrixXd */
    PyArrayObject *A = (PyArrayObject *) X;
    PyArrayObject *B = (PyArrayObject *) Y;
	Map<MatrixXd> _X((double *) PyArray_DATA(A), PyArray_DIMS(A)[0], PyArray_DIMS(A)[1]);
	Map<MatrixXd> _Y((double *) PyArray_DATA(B), PyArray_DIMS(B)[0], PyArray_DIMS(B)[1]);
	grow(_X, _Y);
}
using namespace boost::python;
BOOST_PYTHON_MODULE(TSP) {
    class_<TSP>("TSP", init<double, double, double>())
        .def("grow", &TSP::grow_python)
        .def("regularize", &TSP::regularize)
        .def("emi", &TSP::emi)
        .def("reg_emi", &TSP::reg_emi)
        .def("size", &TSP::size)
        .def("partitions_bounds", &TSP::partitionsBounds)
    ;
}
#endif