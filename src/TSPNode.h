// -*- lsst-c++ -*-
#include <iostream>
#include <numeric>
#include <memory>
#include <vector>
#include <cmath>
#include <chrono>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;
typedef Array<bool,Dynamic,1> ArrayXb;
/**
 * @brief Implementation of a tree-structerd partition (TSP) for an histogram-based mutual information estimator.
 * 
 * @see [Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation](https://sail.usc.edu/publications/files/silva_tit_2012.pdf).
 */
class TSPNode {
    public:
        /// Pointer to left child node.
        TSPNode *left;
        /// Pointer to right child node.
        TSPNode *right;
        /// Pointer to parent node.
        TSPNode *parent;
        /// Empirical conditional joint distribution.
        double condJointDist;
        /// Product of empirical conditional marginal distributions.
        double condMargProd;
        /// Conditional mutual information gain.
        double gain;
        /// Number of samples of current partition $A_1\timesA_2$.
        unsigned int n_samples;
        /// Number of samples of X marginal partition $R^p\timesA_2$.
        unsigned int n_marginal_samples_X;
        /// Number of samples of Y marginal partition $A_1\timesR^q$.
        unsigned int n_marginal_samples_Y;
        /// Array of sample indexes of X marginal partition $R^p\timesA_2$.
        vector<unsigned int> marginalSamplesIdxX;
        /// Array of sample indexes of Y marginal partition $A_1\timesR^q$.
        vector<unsigned int> marginalSamplesIdxY;
        /// Vector of lower bounds of the current partition $A_1\timesA_2$.
        VectorXd lowerBounds;
        /// Vector of upper bounds of the current partition $A_1\timesA_2$.
        VectorXd upperBounds;
        /**
         * @brief Class constructor.
         * 
         * @param parent Parent node pointer.
         */
        TSPNode(TSPNode *parent);
        /**
         * @brief Class destructor.
         */
        ~TSPNode();
        /**
         * @brief Creates children nodes corresponding to subpartitions of the current space recursively according to the data-driven growth stopping criterion.
         * 
         * @param parent Parent node.
         * @param partitionDataIdx Indexes array of samples of the current partition.
         * @param data Samples set.
         * @param lowerBounds Partition lower bounds array.
         * @param upperBounds Partition upper bounds array.
         * @param dimX Number of dimensions of the first sample set (|X|).
         * @param dim Number of dimensions of the complete sample set (|[X,Y]|).
         * @param growThresh Growth stopping threshold.
         * @param projDim Dimension identifier for partition data projection.
         * @return Current partition node pointer.
         * @see Equation (8) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        TSPNode* grow(TSPNode* parent, vector<unsigned int> partitionDataIdx, MatrixXd *data, VectorXd lowerBounds, VectorXd upperBounds, vector<bool> dimIndicator, unsigned int dim, double growThresh, unsigned int projDim);
        /**
         * @brief Calculates the conditional marginal empirical distribution of the partition data projection (P(A1 x R^q) or P(R^p x A2)) over the data dimensions determined by the range between `dimStart` and `dimEnd` integers values.
         * 
         * @param data Samples set.
         * @param lowerBounds Partition lower bounds array.
         * @param upperBounds Partition upper bounds array.
         * @param dimStart Start data dimension index.
         * @param dimEnd End data dimension index.
         * @return Conditional marginal empirical distribution of the current partition.
         */
        //******double conditionalMarginalDist(MatrixXd *data, VectorXd lowerBounds, VectorXd upperBounds, unsigned int dimStart, unsigned int dimEnd);
        /**
         * @brief Calculates the product of the conditional mariginal empirical distributions.
         * 
         * @param data Samples set.
         * @param lowerBounds Partition lower bounds array.
         * @param upperBounds Partition upper bounds array.
         * @param dimX Number of dimensions of the first sample set (|X|).
         * @param dim Number of dimensions of the complete sample set (|[X,Y]|).
         * @param projDim Dimension identifier of the non-trivial term of the conditional marginal product to compute.
         * @return Product of conditional marginal empirical distributions.
         * @see Equation (5) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        double conditionalMarginalProd(MatrixXd *data, VectorXd lowerBounds, VectorXd upperBounds, vector<bool> dimIndicator, unsigned int dim, unsigned int projDim);
        /**
         * @brief Calculates the estimated mutual information of the tree rooted in the current node.
         * 
         * @return Sum of conditional information gain of all the leaves.
         */
        double getEMI();
        /**
         * @brief Calculates the number of leaves of the tree rooted in the current node.
         * 
         * @return Number of leaves.
         */
        unsigned int getSize();
        /**
         * @brief Calcultes the conditional mutual information gain of the current partition.
         * 
         * @return Partition conditional mutual information gain.
         * @see Equation (45) of the paper "Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation".
         */
        double CMIGain();
        /**
         * @brief Set the values of two matrices with the lower and upper bounds of each partition of the TSP.
         * 
         * @param partitionsLowerBounds Matrix of lower bounds for each partition of the TSP.
         * @param partitionsUpperBounds Matrix of upper bounds for each partition of the TSP.
         * @param idx Index of the current partition.
         * @return Nothing.
         */
        void partitionBounds(MatrixXd &partitionsLowerBounds, MatrixXd &partitionsUpperBounds, unsigned int &idx);
        /**
         * @brief Deallocates memory destinated to the current node and all its internal nodes recursively.
         * 
         * @return Nothing.
         */ 
        void destroy();
};