#ifndef STATIC_TREE_H
#define STATIC_TREE_H

#include <math.h>
#include <ostream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "FeatureData.h"
#include "time.h"
/*
 * TODO:
 * 2013.3.3
 * 1.修改label的获得方式，现在的方式是加X，应该是加Y
 * 3.修改非0 feature的计算方式
 * 2.检查内存段获取方式	,防止段错误
 */

using namespace std;

class StaticNode {
	public:
		int feature;
		double split;
		double label, loss;

		int m_infty, m_s;
		double s;
		double l_infty, l_s;
		int n_size;
		double l_abs_infty, l_abs_s;
		double l_sqr_infty, l_sqr_s;

};

class StaticTree {
	public:
		// constructor/destructor
		StaticTree(int depth, int n); // creates a tree of fixed depth
		~StaticTree();

		StaticNode*** layers;

		// construction methods
		void clear();
		void startNextLayer();
		void findBestLocalSplits(FeatureData* data, int k, int numtree, float weight);
		void exchangeBestSplits();
		bool containsSplittingFeature(FeatureData* data);

		// prediction methods
		void updateTrainingPredictions(FeatureData *data, int k,
				double learningrate);
		void updatePredictions_2nd(InstanceData *data, int k,
				double learningrate);
		void updatePredictions(InstanceData *data, int k, double learningrate);

		// output methods
		void printTree(double learningrate);

		// info methods
		void getSplit(int node, int &feature, double &split);
		int getNumNodes();
                void traceFeatureSplit(int level, int i, int feature);


	private:
		int depth, layer, nodes, N;

		int nodesAtDepth(int d);
		void clearNode(StaticNode* node);
		double classifyDataPoint(InstanceData* data, int p);
		void classifyDataPoint_2nd(InstanceData* data, int _layer);
		void updateBestSplits(FeatureData* data, int k, int f, int numtree, float weight);
		void printNode(int level, int i, double learningrate);
};

StaticTree::StaticTree(int depth_, int i_size) {
	// initialize parameters
	depth = depth_;
	layer = 0;
	nodes = 1;
	N =i_size;

	// initialize levels
	layers = new StaticNode**[depth];
	for (int i = 0; i < depth; i++) {
		// initialize layer of nodes
		int n = (int) pow(2.f, (double) i);
		layers[i] = new StaticNode*[n];

		// initialize nodes
		for (int j = 0; j < n; j++) {
			layers[i][j] = new StaticNode();
			StaticNode* node = layers[i][j];
			clearNode(node);
		}
	}
}

StaticTree::~StaticTree() {
	// delete histograms and nodes
	for (int i = 0; i < depth; i++) {
		delete[] layers[i];
		layers[i] = NULL;
	}

	// delete layers
	delete[] layers;
}

void StaticTree::clear() {
	// clear each node
	for (int d = 0; d < depth; d++) {
		int n = nodesAtDepth(d);
		for (int j = 0; j < n; j++) {
			StaticNode* node = layers[d][j];
			clearNode(node);
		}
	}

	// reset layer
	layer = 0;
	nodes = 1;
}

void StaticTree::clearNode(StaticNode* node) {

	node->feature = -1;
	node->split = -1.f;
	node->label = -1.0;
	node->loss = -999999999.0;
	node->n_size = N;
	node->m_infty = 0;
	node->l_infty = 0.0;
	node->l_abs_infty = 0.0;
	node->l_sqr_infty = 0.0;
}

void StaticTree::startNextLayer() {
	// increment layer
	layer++;
	nodes = (int) pow(2.f, ((double) layer));
}

int StaticTree::nodesAtDepth(int d) {
	return (int) pow(2.f, ((double) d));
}

void StaticTree::findBestLocalSplits(FeatureData* data, int k, int numtree, float weight) {
	// recompute counts at nodes
	int class_size = data->getK();
	for (int i = 0; i < data->getN(); i++) {
		int n = data->getNode(i);
		weight = data->hasWeight(i, k)? weight: 1;
		layers[layer][n]->m_infty += 1 * int(weight);
		layers[layer][n]->l_infty += data->getMultiResidual(k, i) * weight;
		layers[layer][n]->l_abs_infty += fabs(data->getMultiResidual(k, i)) * weight;
		layers[layer][n]->l_sqr_infty += -pow(data->getMultiResidual(k, i),
				2.0) * weight ;

	}


	// set labels at nodes
	// (this is redundant for many internal nodes -- 
	// overwriting the existing value with the same value --
	// but necessary for the root node and nodes that have stopped short)
	for (int n = 0; n < nodes; n++) {
		StaticNode* node = layers[layer][n];
		StaticNode* child1 = layers[layer + 1][n * 2];
		StaticNode* child2 = layers[layer + 1][n * 2 + 1];
		if (node->m_infty > 0) {
			double para = (double(class_size-1) / class_size);
			double label = para * node->l_infty
				/ (node->l_abs_infty + node->l_sqr_infty);
			node->label = label;
			child1->label = label;
			child2->label = label;
		}
	}

	// iterate over features and update best splits at each node
	for (int f = 0; f < data->getNumFeatures(); f++) {
		updateBestSplits(data, k, f, numtree, weight);
	}
}

// REVISE
void StaticTree::updateBestSplits(FeatureData* data, int k, int f,
		int numtree, float weight) {
	// compute global feature index
	int globalf = data->globalFeatureIndex(f);
	// reset counts at nodes
	//初始化
	for (int n = 0; n < nodes; n++) {
		StaticNode* node = layers[layer][n];
		node->m_s = 0;
		node->l_s = 0.0;
		node->s = 0.f;
		node->l_abs_s = 0.f;
		node->l_sqr_s = 0.f;

	}

	int sortedfeatures = (*data).sortedfeatures[f].size();

	// iterate over feature  
	// REVISE
	// inverse the iteration
	for (int j = 0; j < sortedfeatures; j++) {
		SparseFeature sf = (*data).sortedfeatures[f][j];
		double v = sf.value;
		int i = sf.i_index;
		weight = data->hasWeight(i, k)? weight: 1;
		int n = data->getNode(i);
		double l = data->getMultiResidual(k, i) * weight;
		StaticNode* node = layers[layer][n];
		if (node->m_s > 0 and node->s > v) {
			double loss_i = pow(node->l_s, 2.0) / (double) node->m_s
				+ pow(node->l_infty - node->l_s, 2.0)
				/ (double) (node->m_infty - node->m_s);
			if (node->loss < 0 or loss_i > node->loss) {
				node->loss = loss_i;
				node->feature = globalf;
				node->split = (node->s + v) / 2.f;
				StaticNode* child1 = layers[layer + 1][2 * n];
				child1->n_size = node->m_s;
				child1->label = node->l_s / (node->l_abs_s + node->l_sqr_s);
				StaticNode* child2 = layers[layer + 1][2 * n + 1];
				child2->n_size = node->m_infty - node->m_s;
				child2->label = (node->l_infty - node->l_s)
					/ ((node->l_abs_infty - node->l_abs_s)
							+ (node->l_sqr_infty - node->l_sqr_s));
			}
		}

		// update variables

		// REVISE
		node->m_s += 1 * int(weight);
		node->l_s += l;
		node->l_abs_s += fabs(l) * weight;
		node->l_sqr_s += -pow(l, 2.0) * weight;
		node->s = v;
		//printf("%d:%d:%f:%f\n",n, node->feature, node->split, node->loss);
	}
	for(int n = 0; n < nodes; n++) {
		StaticNode* node = layers[layer][n];
		if((node->m_s < node->m_infty) & (node->m_s > 0)) {
			double loss_i = pow(node->l_s, 2.0) / (double) node->m_s
				+ pow(node->l_infty - node->l_s, 2.0)
				/ (double)(node->m_infty - node->m_s);
			if (node->loss < 0 or loss_i > node->loss) {
				node->loss = loss_i;
				node->feature = globalf;
				node->split = (node->s + -9999999.f) / 2.f;
				StaticNode* child1 = layers[layer + 1][2 * n];
				child1->n_size = node->m_s;
				child1->label = node->l_s / (node->l_abs_s + node->l_sqr_s);
				StaticNode* child2 = layers[layer + 1][2 * n + 1];
				child2->n_size = node->m_infty - node->m_s;
				child2->label = (node->l_infty - node->l_s)
					/ ((node->l_abs_infty - node->l_abs_s)
							+ (node->l_sqr_infty - node->l_sqr_s));
			}

		}
	}
}

void StaticTree::exchangeBestSplits() {
	// instantiate buffer
	int buffersize = nodes * 5;
	double* buffer = new double[buffersize];

	// write layer of tree to buffer
	for (int n = 0; n < nodes; n++) {
		StaticNode* node = layers[layer][n];
		buffer[n * 5 + 0] = node->loss;
		buffer[n * 5 + 1] = node->feature;
		buffer[n * 5 + 2] = node->split;

		StaticNode* child1 = layers[layer + 1][n * 2];
		buffer[n * 5 + 3] = child1->label;

		StaticNode* child2 = layers[layer + 1][n * 2 + 1];
		buffer[n * 5 + 4] = child2->label;
	}

	// get myid and numprocs
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	int numprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// determine isRoot
	int root = numprocs - 1;
	bool isRoot = (myid == root);

	// exchange buffers
	double* rbuffer = (isRoot ? new double[numprocs * buffersize] : NULL);
	MPI_Gather(buffer, buffersize, MPI_DOUBLE, rbuffer, buffersize, MPI_DOUBLE,
			root, MPI_COMM_WORLD);

	// save best global splits
	if (isRoot)
		for (int n = 0; n < nodes; n++) {
			// reset loss at node and get pointers
			StaticNode* node = layers[layer][n];
			node->loss = -1;
			StaticNode* child1 = layers[layer + 1][n * 2];
			StaticNode* child2 = layers[layer + 1][n * 2 + 1];

			// consider loss from all processors
			for (int p = 0; p < numprocs; p++) {
				int offset = p * buffersize + n * 5;
				double loss = rbuffer[offset + 0];

				// update if better than current
				if (node->loss < 0 or loss > node->loss) {
					node->loss = loss;
					node->feature = (int) rbuffer[offset + 1];
					node->split = (double) rbuffer[offset + 2];
					child1->label = rbuffer[offset + 3];
					child2->label = rbuffer[offset + 4];
				}
			}
		}

	// buffer best global splits
	if (isRoot)
		for (int n = 0; n < nodes; n++) {
			StaticNode* node = layers[layer][n];
			buffer[n * 5 + 0] = node->loss;
			buffer[n * 5 + 1] = node->feature;
			buffer[n * 5 + 2] = node->split;

			StaticNode* child1 = layers[layer + 1][n * 2];
			buffer[n * 5 + 3] = child1->label;

			StaticNode* child2 = layers[layer + 1][n * 2 + 1];
			buffer[n * 5 + 4] = child2->label;
		}

	// broadcast best splits
	MPI_Bcast(buffer, nodes * 5, MPI_DOUBLE, root, MPI_COMM_WORLD);

	// update tree with best global splits
	for (int n = 0; n < nodes; n++) {
		StaticNode* node = layers[layer][n];
		node->loss = buffer[n * 5 + 0];
		node->feature = (int) buffer[n * 5 + 1];
		node->split = (double) buffer[n * 5 + 2];

		StaticNode* child1 = layers[layer + 1][n * 2];
		child1->label = buffer[n * 5 + 3];

		StaticNode* child2 = layers[layer + 1][n * 2 + 1];
		child2->label = buffer[n * 5 + 4];
	}

	// delete buffers
	delete[] buffer;
	delete[] rbuffer;
}

bool StaticTree::containsSplittingFeature(FeatureData* data) {
	int feature;
	double split;
	for (int i = 0; i < nodes; i++) {
		getSplit(i, feature, split);
		if (data->isLocalFeature(feature))
			return true;
	}
	return false;
}

void StaticTree::updateTrainingPredictions(FeatureData *data, int k,
		double learningrate) {
	int N = data->getN();
	for (int i = 0; i < N; i++) {
		int node = data->getNode(i);
		double pred = nanToNum(learningrate * layers[layer][node]->label);
		data->updateMultiPred(k, i, pred);
	}
}

//TODO complete k interation
void StaticTree::updatePredictions(InstanceData *data, int k,
		double learningrate) {
	int N = data->getN();
	for (int i = 0; i < N; i++) {
		double pred = nanToNum(learningrate * classifyDataPoint(data, i));
		data->updateMultiPred(k, i, pred);
	}
}
void StaticTree::updatePredictions_2nd(InstanceData *data, int k, double learningrate) {
	int N = data->getN();
//	for(int i = 0; i<depth-1; i++) {
//		classifyDataPoint_2nd(data, i);
//	}
	for (int i = 0; i < N; i++) {
		int node = data->getNode(i);
		double pred = nanToNum(learningrate * layers[layer][node]->label);
		data->updateMultiPred(k, i, pred);
	}

}
void StaticTree::classifyDataPoint_2nd(InstanceData* data, int _layer) {
	for (int i = 0; i < data->getN(); i++) {
		int f = layers[_layer][data->getNode(i)]->feature;
		double s = layers[_layer][data->getNode(i)]->split;
		data->setNode(i, data->getNode(i)<<1);
		if (f >= 0 and data->getFeature(f, i) < s) { // TODO : eliminate branch
			data->setNode(i,data->getNode(i) | 1U); // node[i] += 1
		}
	}
}
double StaticTree::classifyDataPoint(InstanceData* data, int p) {
	// descend tree
	int node = 0;
	for (int i = 0; i < depth - 1; i++) {
		// get feature and split point
		int f = layers[i][node]->feature;
		double s = layers[i][node]->split;

		// check for valid split, otherwise return
		if (f < 0)
			return layers[i][node]->label;

		// perform split
		node <<= 1; // node *= 2, index of left child
		//smaller be right

		double feature_value = data->getFeature(f,p);
		node |= (feature_value < s); // node += 1, if right child
	}

	// return label of leaf node as prediction
	return layers[depth - 1][node]->label;
}


void StaticTree::printNode(int level, int i, double learningrate) {
	// get node
	StaticNode *node = layers[level][i];

	// print node
	if (level > 0)
		printf(",");
	printf("%d:%f:%f",node->feature, node->split, learningrate * node->label);
	if (node->feature >= 0) { // a splitting node
		printNode(level + 1, 2 * i, learningrate); // print left child
		printNode(level + 1, 2 * i + 1, learningrate); // print right child
	}
}

void StaticTree::printTree(double learningrate) {
	printNode(0, 0, learningrate);
	printf("\n");
}

void StaticTree::traceFeatureSplit(int level, int i, int feature) {
	StaticNode *node = layers[level][i];
	if (feature == node->feature) {
		printf("%f\n", node->split);
	}
	if (node->feature >= 0) {
		traceFeatureSplit(level + 1, 2 * i, feature);
		traceFeatureSplit(level + 1, 2 * i + 1, feature);
	}
} 


void StaticTree::getSplit(int node, int &feature, double &split) {
	feature = layers[layer][node]->feature;
	split = layers[layer][node]->split;
}

int StaticTree::getNumNodes() {
	return nodes;
}

#endif
