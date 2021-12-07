// -*- lsst-c++ -*-
#include "TSPNode.h"

TSPNode::TSPNode(TSPNode* parent) : left(nullptr), right(nullptr), parent(parent), gain(-INFINITY), n_samples(0), n_marginal_samples_X(0), n_marginal_samples_Y(0) {
};

TSPNode::~TSPNode() {
};

TSPNode* TSPNode::grow(TSPNode* parent, vector<unsigned int> partitionDataIdx, MatrixXd *data, VectorXd lowerBounds, VectorXd upperBounds, vector<bool> dimIndicator, unsigned int dim, double growThresh, unsigned int projDim) {
    TSPNode *nodeptr = new TSPNode(parent);
    if (projDim == 0) {
        nodeptr->parent = nodeptr;
        nodeptr->marginalSamplesIdxX = partitionDataIdx;
        nodeptr->marginalSamplesIdxY = partitionDataIdx;
        nodeptr->n_marginal_samples_X = partitionDataIdx.size();
        nodeptr->n_marginal_samples_Y = partitionDataIdx.size();
    }
    nodeptr->lowerBounds = lowerBounds;
    nodeptr->upperBounds = upperBounds;
    nodeptr->n_samples = partitionDataIdx.size();
    nodeptr->condJointDist = (double)nodeptr->n_samples / nodeptr->parent->n_samples;
    nodeptr->condMargProd = nodeptr->conditionalMarginalProd(data, lowerBounds, upperBounds, dimIndicator, dim, projDim);
    if (floor(partitionDataIdx.size() / 2) >= growThresh) {
        /* Sample points projection into the target dimension */
        double *projectedData = new double[partitionDataIdx.size()];
        for (unsigned int i = 0; i < partitionDataIdx.size(); i++) {
            projectedData[i] = (*data)(partitionDataIdx[i], projDim % dim);
        }
        /* Sample points projection sort */
        vector<unsigned int> idx(partitionDataIdx.size());
        iota(idx.begin(), idx.end(), 0);
        nth_element(idx.begin(), idx.begin() + (idx.size() - 1) / 2, idx.end(), [&projectedData](size_t i1, size_t i2) {return projectedData[i1] < projectedData[i2];});
        delete[] projectedData;
        /* Left space partition */ 
        vector<unsigned int> leftDataIdx((partitionDataIdx.size() + 1) / 2);
        for (unsigned int i = 0; i < leftDataIdx.size(); i++) {
            leftDataIdx[i] = partitionDataIdx[idx[i]];
        }
        /* Right space partition */ 
        vector<unsigned int> rightDataIdx(partitionDataIdx.size() / 2);
        for (unsigned int i = 0; i < rightDataIdx.size(); i++) {
            rightDataIdx[i] = partitionDataIdx[idx[i + leftDataIdx.size()]];
        }
        /* Subpartitions bounds */
        VectorXd leftUpperBounds = upperBounds;
        leftUpperBounds(projDim % dim) = (*data)(leftDataIdx.back(), projDim % dim);
        VectorXd rightLowerBounds = lowerBounds;
        rightLowerBounds(projDim % dim) = (*data)(leftDataIdx.back(), projDim % dim); 
        /* Node growth recursion */
        nodeptr->left = this->grow(nodeptr, leftDataIdx, data, lowerBounds, leftUpperBounds, dimIndicator, dim, growThresh, projDim + 1);
        nodeptr->right = this->grow(nodeptr, rightDataIdx, data, rightLowerBounds, upperBounds, dimIndicator, dim, growThresh, projDim + 1);  
        /* Conditional mutual information gain */
        nodeptr->gain = nodeptr->CMIGain() * ((double)nodeptr->n_samples / (*data).rows());
    }
    return nodeptr;
};

double TSPNode::conditionalMarginalProd(MatrixXd *data, VectorXd lowerBounds, VectorXd upperBounds, vector<bool> dimIndicator, unsigned int dim, unsigned int projDim) {
    // TODO: Refactor to avoid duplicated code.
    if (projDim == 0) {
        return 1;
    }
    if (dimIndicator[(projDim - 1) % dim]) {
        this->marginalSamplesIdxY = this->parent->marginalSamplesIdxY;
        this->n_marginal_samples_Y = this->parent->n_marginal_samples_Y;
        vector<unsigned int> vect(this->parent->n_marginal_samples_X);
        this->marginalSamplesIdxX = vect;
        for (unsigned int i = 0; i < this->parent->n_marginal_samples_X; i++) {
            for (unsigned int d = 0; d < dim; d++) {
                if (dimIndicator[d]) {
                    if (((*data)(this->parent->marginalSamplesIdxX[i], d) <= lowerBounds(d)) || ((*data)(this->parent->marginalSamplesIdxX[i], d) > upperBounds(d))) {
                        break;
                    }
                }
                if (d == (dim - 1)) {
                    this->marginalSamplesIdxX[this->n_marginal_samples_X++] = this->parent->marginalSamplesIdxX[i];
                }
            }
        }
        return (double)this->n_marginal_samples_X / this->parent->n_marginal_samples_X;
    }
    else {
        this->marginalSamplesIdxX = this->parent->marginalSamplesIdxX;
        this->n_marginal_samples_X = this->parent->n_marginal_samples_X;
        vector<unsigned int> vect(this->parent->n_marginal_samples_Y);
        this->marginalSamplesIdxY = vect;
        for (unsigned int i = 0; i < this->parent->n_marginal_samples_Y; i++) {
            for (unsigned int d = 0; d < dim; d++) {
                if (!dimIndicator[d]) {
                    if (((*data)(this->parent->marginalSamplesIdxY[i], d) <= lowerBounds(d)) || ((*data)(this->parent->marginalSamplesIdxY[i], d) > upperBounds(d))) {
                        break;
                    }
                }
                if (d == (dim - 1)) {
                    this->marginalSamplesIdxY[this->n_marginal_samples_Y++] = this->parent->marginalSamplesIdxY[i];
                }
            }
        }
        return (double)this->n_marginal_samples_Y / this->parent->n_marginal_samples_Y;
    }
};

double TSPNode::CMIGain() {
    return this->left->condJointDist * log2(this->left->condJointDist / this->left->condMargProd) + this->right->condJointDist * log2(this->right->condJointDist / this->right->condMargProd);
};

double TSPNode::getEMI() {
    if (this->left == NULL && this->right == NULL) {
        return 0;
    }
    else {
        return this->gain + this->left->getEMI() + this->right->getEMI();
    }
};

unsigned int TSPNode::getSize() {
    if (this->left == NULL && this->right == NULL) {
        return 1;
    }
    else {
        return this->left->getSize() + this->right->getSize();
    }
};

void TSPNode::partitionBounds(MatrixXd &partitionsLowerBounds, MatrixXd &partitionsUpperBounds, unsigned int &idx) {
    if (this->left == NULL && this->right == NULL) {
        partitionsLowerBounds.row(idx) = this->lowerBounds;
        partitionsUpperBounds.row(idx) = this->upperBounds;
        idx++;
    }
    else {
        this->left->partitionBounds(partitionsLowerBounds, partitionsUpperBounds, idx);
        this->right->partitionBounds(partitionsLowerBounds, partitionsUpperBounds, idx);
    }
};

void TSPNode::destroy() {
    if (this->left != NULL && this->right != NULL) {
        this->left->destroy();
        this->right->destroy();
    }
    delete this;
};
