#ifndef FEATURE_DATA_H
#define FEATURE_DATA_H

#include <ostream> // TODO eliminate unnecessary includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <ostream>
#include <stdlib.h>
#include <math.h>
#include "metrics.h"
#include "loss.h"

using namespace std;

// REVISE
// store the sparse dataset
// only value != 0 need to be stored
struct SparseFeature {
	int i_index;
	double value;
};

class FeatureData { // represents a training data set distributed among processors feature-wise
	public:
		// constructor/destructor
		FeatureData(int n, int numfeatures_, int isrankingset_, int myid_, int numprocs_);
		FeatureData(int n, int k, int numfeatures_, int isrankingset_, int myid_, int numprocs_);
		~FeatureData();

		// reading and initialization
		bool read(const char* file);
		void initMetrics();
		void sort();

		// manage tuples
		void reset();

		// metrics
		void computeMetrics(double &rmse, double &err, double &ndcg);  // TODO for now, just compute using single-core, no comm

		// queries
		int getN();

		int getNumFeatures();
		int getNumQueries();
		int getNode(int i);
		void setNode(int i, int n);
		double getResidual(int i);
		double getMultiResidual(int k, int i);
		double getFeature(int f, int i);
		double getSortedFeature(int f, int i);
		int getSortedIndex(int f, int i);

		int whoHasFeature(int f);
		bool isLocalFeature(int f);
		int localFeatureIndex(int gf);
		int globalFeatureIndex(int lf);

		// prediction
		void updatePred(int i, double p);
		void updateMultiPred(int k, int i, double p); 
		void updateResiduals();
		void updateMultiResiduals();
		void updateMultiPx();

		//
		void updateSparseValue();
		void initSparseValue();

		//private:
	public:
		// dataset descriptors
		int N; // number of data instances
		int K; // number of class
		int numfeatures; // number of features stored on this processor
		bool isrankingset; // whether qids should be expected in file, also whether to compute ranking metrics
		// int minfeature, maxfeature; // range of feature indices on this processor; mapped from [minf,maxf] to [1,numf]; 0 feature for convenience
		int myid, numprocs;
		int numqueries; // number of queries in the data set

		// static attributes
		int* qid; // query id of each instance

		// REVISE

		vector< vector<SparseFeature> > rawfeatures;

		// REVISE 
		// store the sparse dataset
		vector< vector<SparseFeature> > sortedfeatures;
		vector< vector<int> > sparseindices;

		double* label; // target label value of each instance
		double** multi_label;//target label value of each class and instance

		// level-specific attributes
		int* node; // last node reached in tree, permits constant time classification at each level

		//level-specific attributes(multiple)
		//int** node;//last node reached in tree in

		// prediction attributes(single)
		double* pred; // current cumulative prediction for each instance
		double* residual;  // current cumulative residual for each instance

		// prediction attributes(multiple)
		double** multi_pred;//current cumalative prediction for each class and instance
		double** multi_residual;//current cumalative residual for each class and insance
		double** multi_px;//current P_k(x) for each class and instance

		// metric attributes
		double* idealdcg; // ideal dcg by query

		// methods
		bool processLine(int &linenum, ifstream &input, int i);
		bool parseFeatureValue(string &cfeature, string &cvalue);
		int computeNumFeatures(int nf, int numprocs, int myid);
		// REVISE
		int binarySearch(int f, int i);
};

FeatureData::FeatureData(int n, int numfeatures_, int isrankingset_, int myid_, int numprocs_) {
	// N, numfeatures, minfeature, maxfeature
	N = n;
	numfeatures = computeNumFeatures(numfeatures_, numprocs_, myid_);
	isrankingset = isrankingset_;
	// minfeature = minf;
	// maxfeature = maxf;
	myid = myid_;
	numprocs = numprocs_;
	numqueries = -1;

	// qid: limited init, read from file
	qid = new int[n];

	for (int i = 0; i < numfeatures; i++) {
		sortedfeatures.push_back(vector<SparseFeature>());
		sparseindices.push_back(vector<int>());

		rawfeatures.push_back(vector<SparseFeature>());	
	}
	// label: limited init, read from file
	label = new double[n];

	// node: initialized to 0
	node = new int[n];
	for (int i=0; i<n; i++)
		node[i] = 0;

	// pred: initialized to 0.f
	pred = new double[n];
	for (int i=0; i<n; i++)
		pred[i] = 0.;

	// residual: limited init, initialized to label value
	residual = new double[n];

	// idealdcg: no init, computed after file reading, if isrankingset
	idealdcg = NULL;	
}

FeatureData::FeatureData(int n, int k, int numfeatures_, int isrankingset_, int myid_, int numprocs_){
	N = n;
	K = k;
	isrankingset = isrankingset_;
	numfeatures = computeNumFeatures(numfeatures_, numprocs_, myid_);
	myid = myid_;
	numprocs = numprocs_;
	numqueries = -1;
	qid = new int[n];

	for(int i = 0; i < numfeatures; i++){
		sortedfeatures.push_back(vector<SparseFeature>());
		sparseindices.push_back(vector<int>());

		rawfeatures.push_back(vector<SparseFeature>());

	}	
	multi_label = new double*[k];
	multi_residual = new double*[k];
	multi_px = new double*[K];
	for (int i = 0; i<k; i++) {
		multi_label[i] = new double[n];
		multi_px[i] = new double[n];
		multi_residual[i] = new double[n];
		for (int j = 0; j<n; j++) {
			multi_label[i][j] = 0.0;
			multi_px[i][j] = 0.0;
			multi_residual[i][j] = 0.0;
		
		}
		
	}
	node = new int[n];

	for (int i = 0; i<n; i++)
		node[i] = 0;

	idealdcg = NULL;


}
FeatureData::~FeatureData() {
	// delete all 1-d arrays: qid, label, node, pred, residual, idealdcg
	delete [] qid;
	delete [] label;
	delete [] node;
	delete [] pred;
	delete [] residual;
	delete [] idealdcg;

	/*
	// delete all 2-d arrays: rawfeatures, sortedfeatures, sortedindices
	for (int i=0; i<numfeatures; i++) {
	delete [] rawfeatures[i];
	rawfeatures[i] = NULL;

	delete [] sortedfeatures[i];
	sortedfeatures[i] = NULL;

	delete [] sortedindices[i];
	sortedindices[i] = NULL;
	}
	delete[] rawfeatures;
	delete[] sortedfeatures;
	delete[] sortedindices;
	*/
}

bool FeatureData::read(const char* file) {
	// open file, or return error
	ifstream input(file);
	if (input.fail()) {
		fprintf(stderr, "Error: unable to open training file %s\n", file);
		return false;
	}

	// process all data instances
	int linenum = 0;
	for (int i=0; i<N; i++) {
		bool success = processLine(linenum, input, i);
		if (not success) {
			fprintf(stderr, "Error: unable to load training file %s, line %d\n", file, linenum);
			return false;
		}
	}
	// indicate success
	return true;
}

bool FeatureData::processLine(int &linenum, ifstream &input, int i) {
	// setup for reading
	int cqid = -1;
	double clabel = -1;
	string cfeature, cvalue;

	// read line
	string strline;
	getline(input, strline);
	linenum++;

	// check for errors
	if (input.eof()) {
		fprintf(stderr, "Error: end-of-file reached before expected number of training examples were read\n");
		return false;
	} else if (input.fail()) {
		fprintf(stderr, "Error: failure while reading training example\n");
		return false;
	}

	// setup for tokenizing
	char* line = strdup(strline.c_str());
	char* tok = NULL;

	// extract label (first item) and check
	if (not (tok = strtok(line, " "))) {
		fprintf(stderr, "Error: malformed line in training file, missing label\n");
		return false;
	}
	clabel = atof(tok);
	//label[i] = clabel;
	multi_label[int(clabel)][i] = 1;
	// get qid, or ignore if not isrankingset
	string qidstr ("qid");
	if (isrankingset) {
		if (not parseFeatureValue(cfeature, cvalue) or qidstr.compare(cfeature) != 0 or cvalue.empty()) {
			fprintf(stderr, "Error: malformed line in training file, missing qid\n");
			return false;
		}
		cqid = atoi(cvalue.c_str());
		qid[i] = cqid;
		if (not parseFeatureValue(cfeature, cvalue)) return true;
		if (cvalue.empty()) {
			fprintf(stderr, "Error: invalid feature/value pair in training file\n");
			return false;
		}
	} else {
		if (not parseFeatureValue(cfeature, cvalue)) return true;
		if (cvalue.empty()) {
			fprintf(stderr, "Error: invalid feature/value pair in training file\n");
			return false;
		}
		if (qidstr.compare(cfeature)) { // qid present
			if (not parseFeatureValue(cfeature, cvalue)) return true;
			if (cvalue.empty()) {
				fprintf(stderr, "Error: invalid feature/value pair in training file\n");
				return false;
			}
		}
	}

	// get feature values
	int feature = -1;
	double value = -1.f;
	do {
		// validate pair
		if (cvalue.empty()) {
			fprintf(stderr, "Error: invalid feature/value pair in training file\n");
			return false;
		}

		// record feature
		feature = atoi(cfeature.c_str());
		// if (feature > maxfeature) break;
		// else if (feature >= minfeature) {
		//     value = (double) atof(cvalue.c_str());
		//     rawfeatures[feature-minfeature][i] = value;
		// }
		if (isLocalFeature(feature)) {
			value = (double) atof(cvalue.c_str());
			int lf = localFeatureIndex(feature);
			if (lf > numfeatures) {
				fprintf(stderr, "Error: feature index %d out of expected range\n", feature);
				return false;
			} else {	
				SparseFeature sf;
				sf.i_index = i;
				sf.value = value;

				SparseFeature rf;
				rf.i_index = i;
				rf.value = value;

				rawfeatures[lf].push_back(rf);

				sortedfeatures[lf].push_back(sf);
				sparseindices[lf].push_back(i);
			} 
		}
	} while (parseFeatureValue(cfeature, cvalue));

	// clean up
	free(line);
	// return
	return true;
}

bool FeatureData::parseFeatureValue(string &cfeature, string &cvalue) {
	// get token
	char* tok;
	if (not (tok = strtok(NULL, " \n"))) // line stored in state from previous call to strtok
		return false;

	// find colon
	string bit = tok;
	int colon_index = bit.find(":");

	// return empty value if colon is missing
	if (colon_index == bit.npos){
		cfeature = bit;
		cvalue = string();
		return true;
	}

	// split string
	cfeature = bit.substr(0, colon_index);
	cvalue = bit.substr(colon_index+1,bit.length()-colon_index-1);

	return true;
}

void FeatureData::reset() {
	// clear nodes before next tree
	for (int i=0; i<N; i++) {
		node[i] = 0;
	}
}

class FeatureValuePair {
	public:
		int index;
		double value;
};

struct CompareFeatureValuePairs {
	bool operator() (FeatureValuePair* fv1, FeatureValuePair* fv2) {
		return (fv1->value < fv2->value);
	}
};

struct VectorSortP {
	bool operator() (const SparseFeature a, const SparseFeature b) const {
		return (a.value < b.value);
	}
};
void FeatureData::sort() {
	for (int f = 0; f < sortedfeatures.size(); f++) { 
		std::sort(sortedfeatures[f].begin(), sortedfeatures[f].end(), VectorSortP());					
	}
}


void FeatureData::initMetrics() {
	// if not ranking data set, return
	if (not isrankingset) return;

	// compute number of queries and initialize idealdcg
	numqueries = computeNumQueries(N, qid);
	idealdcg = new double[numqueries];

	// compute idealdcg for each query
	computeIdealDCG(N, qid, label, idealdcg);
}

void FeatureData::computeMetrics(double &rmse, double &err, double &ndcg) {
	// compute rmse
	rmse = sqrt(computeBoostingSE(N, label, pred) / (double) N);

	// if not ranking data set, return
	if (not isrankingset) return;

	// compute ranking metrics
	double rawerr, rawndcg;
	computeBoostingRankingMetrics(N, qid, pred, label, idealdcg, rawerr, rawndcg);
	err = rawerr / (double) numqueries;
	ndcg = rawndcg / (double) numqueries;
}

int FeatureData::getN() {
	return N;
}

int FeatureData::getNumFeatures() {
	return numfeatures;
}

int FeatureData::getNumQueries() {
	return numqueries;
}

// int FeatureData::getMinFeature() {
//     return minfeature;
// }

int FeatureData::whoHasFeature(int f) {
	return f % numprocs;
}

bool FeatureData::isLocalFeature(int f) {
	return whoHasFeature(f) == myid;
}

int FeatureData::localFeatureIndex(int gf) {
	return gf / numprocs;
}

int FeatureData::globalFeatureIndex(int lf) {
	return lf * numprocs + myid;
}

int FeatureData::computeNumFeatures(int nf, int numprocs, int myid) {
	return (nf / numprocs) + (nf % numprocs > myid);
}

int FeatureData::binarySearch(int f, int i) {
	int first, last, mid = 0;
	first = 0;
	last = rawfeatures[f].size() - 1;
	bool found = false;
	while((!found) && (first <= last)) {
		mid = first + (last - first) / 2;
		if (rawfeatures[f][mid].i_index == i)
			found = true;
		else if (i < rawfeatures[f][mid].i_index) 
			last = mid - 1;
		else if (i > rawfeatures[f][mid].i_index)
			first = mid + 1;
	}

	if (found)
		return mid;

	return -1;
}

int FeatureData::getNode(int i) {
	return node[i];
}

void FeatureData::setNode(int i, int n) {
	node[i] = n;
}

double FeatureData::getResidual(int i) {
	return residual[i];
}

double FeatureData::getMultiResidual(int k, int i){
	return multi_residual[k][i];
}


double FeatureData::getFeature(int f, int i) {	
	// binarySearch needs log(rawfeatures[f].size())
	int index = binarySearch(f, i);
	if (index == -1)
		return 0.0;

	return rawfeatures[f][index].value;
}

double FeatureData::getSortedFeature(int f, int i) {
	if (i >= N -  sortedfeatures[f].size())
		return sortedfeatures[f][i - (N - sortedfeatures[f].size())].value;
	return 0.f;
}

int FeatureData::getSortedIndex(int f, int i) {
	if (i >= N - sortedfeatures[f].size())
		return sortedfeatures[f][i - (N - sortedfeatures[f].size())].i_index;

	return -1;
}

void FeatureData::updatePred(int i, double p) {
	pred[i] += p;
}

//FIXME
void FeatureData::updateMultiPred(int k, int i, double p) {
	multi_pred[k][i] += p;
}

void FeatureData::updateMultiResiduals() {

	computeMultiGradient(N, K, multi_label, multi_px, qid, multi_residual);
	for (int i = 0; i<N; i++)
		node[i] = 0;

}

void FeatureData::updateResiduals() {

	computegradient(N, label, pred, qid, residual);

	for (int i = 0; i < N; i++)
		node[i] = 0;
}

//TODO MPI processing
void FeatureData::updateMultiPx() {
	double* temp = new double[N];
	for (int i=0; i<N; i++) {
		temp[i] = 0;
		for (int k=0; k<K; k++) {
			temp[i] += multi_pred[k][i];		
		}
		for (int k=0; k<K; k++) {
			multi_px[k][i] = multi_pred[k][i]/temp[i];
		}
	}
}
#endif
