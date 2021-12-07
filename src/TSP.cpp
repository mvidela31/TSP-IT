// -*- lsst-c++ -*-
#include "TSP.h"

TSP::TSP(double l_bn, double w_bn, double lambda) : l_bn(l_bn), w_bn(w_bn), lambda(lambda), root(nullptr) {
    if (l_bn >= 1/3.0 || l_bn <= 0) {
        throw invalid_argument("Parameter `l_bn` must belong to the interval (0, 1/3).");
    }
};

TSP::~TSP() {
    if (this->root != NULL) {
        this->root->destroy();
    }
};

void TSP::grow(MatrixXd X, MatrixXd Y) {
    if (this->root != NULL) {
        this->root->destroy();
    }
    unsigned int dimX = X.cols();
    MatrixXd *data = new MatrixXd(X.rows(), X.cols() + Y.cols());

    unsigned int j = 0;
    vector<bool> dimIndicator((*data).cols());
    for (unsigned int i = 0; i < max(X.cols(), Y.cols()); i++) {
        if (i < X.cols()) {
            (*data).col(j) = X.col(i);
            dimIndicator[j] = true;
            j++;
        }
        if (i < Y.cols()) {
            (*data).col(j) = Y.col(i);
            j++;
        }
    }
    this->dim = (*data).cols();
    this->n_samples = (*data).rows();
    vector<unsigned int> partitionDataIdx(this->n_samples);
    iota(partitionDataIdx.begin(), partitionDataIdx.end(), 0);
    double growThresh = ceil(this->w_bn * pow(this->n_samples, 1 - this->l_bn));
    this->root = this->root->grow(nullptr, partitionDataIdx, data, (*data).colwise().minCoeff().array() - 0.001, (*data).colwise().maxCoeff().array() + 0.001, dimIndicator, this->dim, growThresh, 0);
    this->tsp_emi = this->root->getEMI();
    this->tsp_reg_emi = this->tsp_emi;
    this->tsp_size = this->root->getSize();
    delete data;
};

void TSP::regularize() {
    if (this->root == NULL) {
        throw invalid_argument("Observations not provided.");
    }
    unsigned int optimalSize = 1;
    unsigned int fullTreeSize = this->root->getSize();
    double bn = this->w_bn * pow(this->n_samples, -this->l_bn);
    double inv_deltan = exp(pow(this->n_samples, 1 / 3.0));
    double optimalEMIArray[fullTreeSize] = {};
    TSPNode* internalNodesArray[fullTreeSize] = {};
    this->minimumCostTrees(internalNodesArray, optimalEMIArray);
    double cost, regularizer;
    double minCost = -optimalEMIArray[0];
    for (unsigned int k = 2; k <= fullTreeSize; k++) {
        regularizer = (12 / bn) * sqrt((log(8 * inv_deltan) + k * ((this->dim + 1) * log(2) + this->dim * log(this->n_samples))) * (8.0 / this->n_samples));
        cost = -optimalEMIArray[k-1] + this->lambda * regularizer;
        if (cost < minCost) {
            minCost = cost;
            optimalSize = k;
        }
    }
    this->tsp_emi = optimalEMIArray[optimalSize-1];
    this->tsp_reg_emi = -minCost;
    this->tsp_size = optimalSize;
};

void TSP::minimumCostTrees(TSPNode **internalNodesArray, double *optimalEMIArray) {
    unsigned int i;
    TSPNode *node;
    internalNodesArray[0] = this->root;
    for (unsigned int k = 0; k < this->root->getSize() - 1; k++) {
        i = 0;
        optimalEMIArray[k+1] = optimalEMIArray[k] + internalNodesArray[k]->gain;
        node = internalNodesArray[k];
        if (node->left != NULL) {
            this->subadditiveInsert(node->left, internalNodesArray, k+i);
            i++;
        }
        if (node->right != NULL) {
            this->subadditiveInsert(node->right, internalNodesArray, k+i);
        }
    }
};

void TSP::subadditiveInsert(TSPNode* node, TSPNode** internalNodesArray, unsigned int size) {
    TSPNode *tmp;
    internalNodesArray[size] = node;
    for (unsigned int idx = size; idx > 0; idx--) {
        if (internalNodesArray[idx-1]->left == NULL && internalNodesArray[idx-1]->right == NULL) {
            break;
        }
        if (internalNodesArray[idx-1]->gain > internalNodesArray[idx]->gain) {
            tmp = internalNodesArray[idx];
            internalNodesArray[idx] = internalNodesArray[idx-1];
            internalNodesArray[idx-1] = tmp;
        }
    }
};

double TSP::emi() {
    if (this->root == NULL) {
        throw invalid_argument("Observations not provided.");
    }
    return this->tsp_emi;
};

double TSP::reg_emi() {
    if (this->root == NULL) {
        throw invalid_argument("Observations not provided.");
    }
    return this->tsp_reg_emi;
};

unsigned int TSP::size() {
    if (this->root == NULL) {
        throw invalid_argument("Observations not provided.");
    }
    return this->tsp_size;
};

void TSP::partitionsBounds() {
    // TODO: Wrap the Eigen matrices with partition bounds to Python NumPy arrays.
    unsigned int idx = 0;
    MatrixXd lowerBounds(this->tsp_size, this->dim);
    MatrixXd upperBounds(this->tsp_size, this->dim);
    root->partitionBounds(lowerBounds, upperBounds, idx);
    std::cout << lowerBounds << "\n" << std::endl;
    std::cout << upperBounds << std::endl;
};
