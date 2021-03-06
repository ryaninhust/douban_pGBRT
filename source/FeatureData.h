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
		FeatureData(int n, int k, int numfeatures_, int isrankingset_, int myid_,
				int numprocs_);
		~FeatureData();

		// reading and initialization
		bool read(const char* file);
		void initMetrics();
		void sort();

		// manage tuples
		void reset();

		// metrics
		void computeMetrics(double &rmse, double &err, double &ndcg, double &rate, double &loss); // TODO for now, just compute using single-core, no comm

		// queries
		int getN();
		int getK();

		int getNumFeatures();
		int getNumQueries();
		int getNode(int i);
		void setNode(int i, int n);
		double getResidual(int i);
		double getMultiResidual(int k, int i);
		double getFeature(int f, int i);
		double getSortedFeature(int f, int i);
		int getSortedIndex(int f, int i);
		bool hasWeight(int i, int k);

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
		void predResult(string filePath);

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
		double* bincounts;
		// REVISE

		vector<vector<SparseFeature> > rawfeatures;

		// REVISE
		// store the sparse dataset
		vector<vector<SparseFeature> > sortedfeatures;
		vector<vector<int> > sparseindices;

		double** multi_label; //target label value of each class and instance

		// level-specific attributes
		int* node; // last node reached in tree, permits constant time classification at each level

		//level-specific attributes(multiple)
		//int** node;//last node reached in tree in

		// prediction attributes(single)

		// prediction attributes(multiple)
		double** multi_pred; //current cumalative prediction for each class and instance
		double** multi_residual; //current cumalative residual for each class and insance
		double** multi_px; //current P_k(x) for each class and instance

		// metric attributes
		double* idealdcg; // ideal dcg by query

		// methods
		bool processLine(int &linenum, ifstream &input, int i);
		bool parseFeatureValue(string &cfeature, string &cvalue);
		int computeNumFeatures(int nf, int numprocs, int myid);
		// REVISE
		int binarySearch(int f, int i);
};

FeatureData::FeatureData(int n, int k, int numfeatures_, int isrankingset_,
		int myid_, int numprocs_) {
	N = n;
	K = k;
	isrankingset = isrankingset_;
	numfeatures = computeNumFeatures(numfeatures_, numprocs_, myid_);
	myid = myid_;
	numprocs = numprocs_;
	numqueries = -1;
	qid = new int[n];

	for (int i = 0; i < numfeatures; i++) {
		sortedfeatures.push_back(vector<SparseFeature>());
		sparseindices.push_back(vector<int>());
		rawfeatures.push_back(vector<SparseFeature>());
	}
	multi_label = NULL;
	multi_residual = NULL;
	multi_px = NULL;
	multi_pred = NULL;
	multi_label = new double*[k];
	multi_residual = new double*[k];
	multi_px = new double*[k];
	multi_pred = new double*[k];
	for (int i = 0; i < k; i++) {
		multi_pred[i] = new double[n];
		multi_label[i] = new double[n];
		multi_px[i] = new double[n];
		multi_residual[i] = new double[n];
		for (int j = 0; j < n; j++) {
			multi_label[i][j] = 0.0;
			multi_px[i][j] = 0.0;
			multi_pred[i][j] = 0.0;
			multi_residual[i][j] = 0.0;

		}

	}
	node = new int[n];
	bincounts = new double[k];

	for (int j = 0; j < k; j++)
		bincounts[j] = 0;
	for (int i = 0; i < n; i++)
		node[i] = 0;	
	idealdcg = NULL;

}
FeatureData::~FeatureData() {
	// delete all 1-d arrays: qid, label, node, pred, residual, idealdcg
	delete[] qid;
	delete[] node;
	delete[] idealdcg;


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
	for (int i = 0; i < N; i++) {
		bool success = processLine(linenum, input, i);
		if (not success) {
			fprintf(stderr, "Error: unable to load training file %s, line %d\n",
					file, linenum);
			return false;
		}
	}
	// indicate success
  	for (int j = 0; j < K; j++) {
  		bincounts[j] = bincounts[j] / linenum;
  	}
  	//init multi_pred
    /*
  	for (int i =0; i < N; i++) {
  		for (int j = 0; j < K; j++) {
  			multi_pred[j][i] = bincounts[j];
  		}
  	}
    */
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
		fprintf(
				stderr,
				"Error: end-of-file reached before expected number of training examples were read\n");
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
		fprintf(stderr,
				"Error: malformed line in training file, missing label\n");
		return false;
	}
	clabel = atof(tok);
	multi_label[int(clabel) - 1][i] = 1;
	bincounts[int(clabel) - 1] += 1;
	// get qid, or ignore if not isrankingset
	string qidstr("qid");
	if (isrankingset) {
		if (not parseFeatureValue(cfeature, cvalue)
				or qidstr.compare(cfeature) != 0 or cvalue.empty()) {
			fprintf(stderr,
					"Error: malformed line in training file, missing qid\n");
			return false;
		}
		cqid = atoi(cvalue.c_str());
		qid[i] = cqid;
		if (not parseFeatureValue(cfeature, cvalue))
			return true;
		if (cvalue.empty()) {
			fprintf(stderr,
					"Error: invalid feature/value pair in training file\n");
			return false;
		}
	}
	else {
		if (not parseFeatureValue(cfeature, cvalue))
			return true;
		if (cvalue.empty()) {
			fprintf(stderr,
					"Error: invalid feature/value pair in training file\n");
			return false;
		}
		if (not qidstr.compare(cfeature)) { // qid present
			if (not parseFeatureValue(cfeature, cvalue))
				return true;
			if (cvalue.empty()) {
				fprintf(stderr,
						"Error: invalid feature/value pair in training file\n");
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
			fprintf(stderr,
					"Error: invalid feature/value pair in training file\n");
			return false;
		}

		feature = atoi(cfeature.c_str()) - 1;
		// if (feature > maxfeature) break;
		// else if (feature >= minfeature) {
		//     value = (double) atof(cvalue.c_str());
		//     rawfeatures[feature-minfeature][i] = value;
		// }
		if (isLocalFeature(feature)) {
			value = (double) atof(cvalue.c_str());
			int lf = localFeatureIndex(feature);
			if (lf > numfeatures) {
				fprintf(stderr,
						"Error: feature index %d out of expected range\n",
						feature);
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
	if (colon_index == bit.npos) {
		cfeature = bit;
		cvalue = string();
		return true;
	}

	// split string
	cfeature = bit.substr(0, colon_index);
	cvalue = bit.substr(colon_index + 1, bit.length() - colon_index - 1);

	return true;
}

void FeatureData::reset() {
	// clear nodes before next tree
	for (int i = 0; i < N; i++) {
		node[i] = 0;
	}
}

class FeatureValuePair {
	public:
		int index;
		double value;
};

struct CompareFeatureValuePairs {
	bool operator()(FeatureValuePair* fv1, FeatureValuePair* fv2) {
		return (fv1->value < fv2->value);
	}
};
//修改比较大小，现在的排列是从大到小
struct VectorSortP {
	bool operator()(const SparseFeature a, const SparseFeature b) const {
		return (a.value > b.value);
	}
};
void FeatureData::sort() {
	for (int f = 0; f < sortedfeatures.size(); f++) {
		std::sort(sortedfeatures[f].begin(), sortedfeatures[f].end(),
				VectorSortP());
	}
}

void FeatureData::initMetrics() {
	// if not ranking data set, return
	if (not isrankingset)
		return;

	// compute number of queries and initialize idealdcg
	numqueries = computeNumQueries(N, qid);
	idealdcg = new double[numqueries];

	// compute idealdcg for each query
	//	computeIdealDCG(N, qid, label, idealdcg);
}

void FeatureData::computeMetrics(double &rmse, double &err, double &ndcg, double &rate, double &loss) {
	// compute rmse
	rmse = sqrt(
			computeMultiBoostingSE(N, K, multi_label, multi_px) / (double) N * K);
    rate = computeRightSize(N, K, multi_label, multi_px) / (double)N;
    loss = computeLogLoss(N, K, multi_label, multi_px);

	// if not ranking data set, return
	if (not isrankingset)
		return;

	// compute ranking metrics
	double rawerr, rawndcg;
	//	computeBoostingRankingMetrics(N, qid, pred, label, idealdcg, rawerr, rawndcg);
	err = rawerr / (double) numqueries;
	ndcg = rawndcg / (double) numqueries;
}

int FeatureData::getN() {
	return N;
}

int FeatureData::getK() {
	return K;
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
	while ((!found) && (first <= last)) {
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

double FeatureData::getMultiResidual(int k, int i) {
	return multi_residual[k][i];
}

bool FeatureData::hasWeight(int i, int k) {
	return multi_label[k][i] == 1;
}
double FeatureData::getFeature(int f, int i) {
	// binarySearch needs log(rawfeatures[f].size())
	int index = binarySearch(f, i);
	if (index == -1)
		return -9999999.f;

	return rawfeatures[f][index].value;
}

double FeatureData::getSortedFeature(int f, int i) {
	if (i >= N - sortedfeatures[f].size())
		return sortedfeatures[f][i - (N - sortedfeatures[f].size())].value;
	return -9999999.f;
}

int FeatureData::getSortedIndex(int f, int i) {
	if (i >= N - sortedfeatures[f].size())
		return sortedfeatures[f][i - (N - sortedfeatures[f].size())].i_index;

	return -1;
}
/*
void FeatureData::updateMultiPred(int k, int i, double p) {
	multi_pred[k][i] = nanToNum(multi_pred[k][i] + p);
	double offset = 0.0;
	if (multi_pred[k][i] > 700.0) {
		printf("hit");
		multi_pred[k][i] = 600.0;
		offset = multi_pred[k][i] - 600.0;
		for (int j = 0; j < K; j++) {
			if(j!= k) {
				multi_pred[j][i] = multi_pred[j][i] - offset;
			}
		}
	}

}
*/

void FeatureData::updateMultiPred(int k, int i, double p) {
	multi_pred[k][i] = multi_pred[k][i] + p;
}

void FeatureData::updateMultiResiduals() {

	computeMultiGradient(N, K, multi_label, multi_px, qid, multi_residual);
	for (int i = 0; i < N; i++)
		node[i] = 0;

}

//TODO MPI processing
/*
void FeatureData::updateMultiPx() {
	//double* temp = new double[N];
	for (int i = 0; i < N; i++) {
		double temp = 0.0;
        int flag = 0;
		for (int k = 0; k < K; k++) {
		    if(isinf(exp(multi_pred[k][i])) or multi_pred[k][i]==1.79769313e+308) {
		    	multi_px[k][i] = 1.0;
		    	flag = 1;
		    	for(int j = 0; j < K; j++){
		    		if(j != k) {
		    			multi_px[j][i] = 0.0;
		    		}
		    	}
		    	break;
		    }
			//temp =nanToNum(temp + nanToNum(exp(multi_pred[k][i])));
		    double a = exp(100);
		    temp = temp + exp(multi_pred[k][i]);
		}
		if(flag) {
			continue;
		}
		for (int k = 0; k < K; k++) {
			//multi_px[k][i] = nanToNum(exp(multi_pred[k][i])) / temp;
			multi_px[k][i] = exp(multi_pred[k][i]) / temp;
		}
	}
}
*/
void FeatureData::updateMultiPx() {
	//double* temp = new double[N];
	for (int i = 0; i < N; i++) {
		double temp = 0.0;
		for (int k = 0; k < K; k++) {
			temp = nanToNum(temp + nanToNum(exp(multi_pred[k][i])));
			}
		for (int k = 0; k < K; k++) {
			multi_px[k][i] = nanToNum(nanToNum(exp(multi_pred[k][i])) / temp);
		}
	}
}
void FeatureData::predResult(string filePath) {
//TODO maybe extract to args
        ofstream outPutFile;
	outPutFile.open(filePath.c_str());
	for (int i = 0; i < N; i++) {
		double max = multi_px[0][i];
		int r_label = 0;
		int r_pred = 0;
		for (int k = 0; k < K; k++) {
			if (multi_label[k][i] == 1) {
				r_label = k;
			}
			if (multi_px[k][i]  > max) {
				max = multi_px[k][i];
				r_pred = k;
			}
		}
		outPutFile << r_label << r_pred <<"\n";
		//printf("%d %d\n",r_label, r_pred);
	}
	outPutFile.close();
}
#endif
