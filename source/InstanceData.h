#ifndef INSTANCE_DATA_H
#define INSTANCE_DATA_H

#include <ostream> // TODO eliminate unnecessary includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <ostream>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "metrics.h"
#include "loss.h"
#include "FeatureData.h"

using namespace std;

class InstanceData { // represents a test data set distributed among processors instance-wise
public:
	// constructor/destructor
	InstanceData(int n, int k, int numfeatures_, bool isrankingset_, int mini,
			int maxi);
	~InstanceData();

	// reading and initialization
	bool read(const char* file, int filesize, double* bincounts);
	void initMetrics();

	// metrics
	void computeMetrics(double &rmse, double &err, double &ndcg, double &rate, double &loss);

	// queries
	int getN();
	int getNumFeatures();
	int getNumQueries();
	double getFeature(int f, int i);

	// prediction
	void updatePred(int i, double p);
	void updateMultiPred(int k, int i, double p);
	void updateMultiPx();
	void predResult(string filePath);
	void setNode(int i, int n);
	int getNode(int i);
	void reset();

private:
	// dataset descriptors
	int N; // number of data instances
	int K; // number of class
	double* bincounts;
	int* node;
	int numfeatures; // number of features stored on this processor
	bool isrankingset; // whether qids should be expected in file, also whether to compute ranking metrics
	int numqueries; // number of queries in the data set
	int minindex, maxindex;

	// static attributes
	vector<vector<SparseFeature> > rawfeatures;
	int* qid; // query id of each instance
	vector<int>* qidtemp;
	//double* label; // target label value of each instance
	vector<double>* labeltemp;

	// prediction attributes
	//double* pred; // current cumulative prediction for each instance
	double** multi_pred; //current cumlative prediction for each class k and each instance
	double** multi_label; // add
	double** multi_px;
	// metric attributes
	double* idealdcg; // ideal dcg by query

	// methods
	bool processLineHeader(int &linenum, ifstream &input, char* &line,
			double &label, int &qid);
	bool storeLine(char* line, int i, double label, int qid);
	int binarySearch(int f, int i);
	bool parseFeatureValue(string &cfeature, string &cvalue);
};

InstanceData::InstanceData(int n, int k, int numfeatures_, bool isrankingset_,
		int mini, int maxi) {
	// N, numfeatures
	N = n;
	K = k;
	numfeatures = numfeatures_;
	isrankingset = isrankingset_;
	numqueries = -1;
	minindex = mini;
	maxindex = maxi;

	// qid: limited init, read from file
	qid = NULL;
	qidtemp = new vector<int>(n, 0);
	node = NULL;
	node = new int[n];
	for(int i = 0; i < n; i++) {
		node[i] = 0;
	}

	// features: limited init, defaulted to minimum value (for missing values) and read from file
	for (int i = 0; i < numfeatures; i++)
		rawfeatures.push_back(vector<SparseFeature>());

	// label: limited init, read from file
	//label = NULL;
	labeltemp = new vector<double>(n, 0.f);

	// pred: initialized to 0.f
	//pred = NULL;
	multi_label = new double*[k];
	multi_pred = new double*[k];
	multi_px = new double*[k];
	for (int i = 0; i < k; i++) {
		multi_label[i] = new double[n];
		multi_pred[i] = new double[n];
		multi_px[i] = new double[n];
		for (int j = 0; j < N; j++) {
			multi_label[i][j] = 0.0;
			multi_pred[i][j] = 0.0;
			multi_px[i][j] = 0.0;
		}
	}

	// idealdcg: no init, computed after file reading, if isrankingset
	idealdcg = NULL;
}

InstanceData::~InstanceData() {
	delete[] qid;
	delete qidtemp;
	//delete [] label;
	delete labeltemp;
	//delete [] pred;
	delete[] idealdcg;

//TODO
//增加向量析构
}

bool InstanceData::read(const char* file, int filesize, double* bincounts) {
	// open file, or return error
	ifstream input(file);
	if (input.fail()) {
		fprintf(stderr, "Error: unable to open validation/test file %s\n",
				file);
		return false;
	}

	// track line number for error messages
	int linenum = 0;

	// skip to my section
	string strline;
	for (int i = 0; i < minindex - 1; i++) {
		getline(input, strline); // skip to section (minindex-1, minindex+N)
		linenum++;
		if (input.eof()) {
			fprintf(
					stderr,
					"Error: end-of-file reached before expected number of validation/test examples were read.\n");
			fprintf(stderr, "Error: unable to load validation/test file %s\n",
					file);
			return false;
		} else if (input.fail()) {
			fprintf(stderr,
					"Error: failure while reading validation/test example.\n");
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}
	}

	// variables
	char* line = NULL;
	double currlabel;
	int prevqid = -1, currqid;

	// get previous qid
	if (minindex > 0)
		if (not processLineHeader(linenum, input, line, currlabel, prevqid)) { // just read the qid, don't store anything
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}
	free(line);

	// skip the remainder of previous query and store allotted data instances
	int j = 0;
	for (int i = 0; i < N; i++) {
		// get label and qid
		if (not processLineHeader(linenum, input, line, currlabel, currqid)) {
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}

		// if ranking set and still previous query, continue
		if (isrankingset and currqid == prevqid) {
			free(line);
			continue;
		}
		// otherwise store line
		bool success = storeLine(line, j++, currlabel, currqid);
		if (not success) {
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}
	}

	// continue until the entire final query has been read
	int finalqid = currqid;
	while (isrankingset and maxindex + 1 < filesize) {
		// get label and qid
		if (not processLineHeader(linenum, input, line, currlabel, currqid)) {
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}

		// done if next query
		if (currqid != finalqid) {
			free(line);
			break;
		}

		// allocate space for new instance
		if (j >= N) {
//			for (int f = 0; f < numfeatures; f++)
//				features[f]->push_back(0.f);
			qidtemp->push_back(0);
			labeltemp->push_back(0.f);
		}

		// store line
		bool success = storeLine(line, j++, currlabel, currqid);
		if (not success) {
			fprintf(stderr,
					"Error: unable to load validation/test file %s, line %d\n",
					file, linenum);
			return false;
		}
	}

	// update N
	N = j;

	// convert qid to array
	qid = new int[N];
	for (int i = 0; i < N; i++)
		qid[i] = qidtemp->at(i);
	delete qidtemp;
	qidtemp = NULL;
	for (int i = 0; i < N; i++) {
		// 这里-1
		multi_label[int(labeltemp->at(i)) - 1][i] = 1.0;
	}
/*
  	for (int k = 0; k < K; k++) {
  		for (int i =0; i < N; i++) {
  			multi_pred[k][i] = bincounts[k];
  		}
  	}
*/
	delete labeltemp;
	labeltemp = NULL;

	// indicate success
	return true;
}

bool InstanceData::processLineHeader(int &linenum, ifstream &input, char* &line,
		double &label, int &qid) {
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
				"Error: end-of-file reached before expected number of validation/test examples were read\n");
		return false;
	} else if (input.fail()) {
		fprintf(stderr,
				"Error: failure while reading validation/test example\n");
		return false;
	}

	// convert line
	line = strdup(strline.c_str());
	char* tok = NULL;

	// extract label
	if (not (tok = strtok(line, " "))) {
		fprintf(
				stderr,
				"Error: malformed line in validation/test file, missing label\n");
		return false;
	}
	label = atof(tok);

	// extract qid
	if (isrankingset) {
		string qidstr("qid");
		if (not parseFeatureValue(cfeature, cvalue)
				or qidstr.compare(cfeature) != 0 or cvalue.empty()) {
			fprintf(
					stderr,
					"Error: malformed line in validation/test file, missing qid\n");
			return false;
		}
		qid = atoi(cvalue.c_str());	// store label and qid

	}

	// return	// store label and qid

	return true;
}

bool InstanceData::storeLine(char* line, int i, double label, int qid) { // uses tokenizer from call to processLineHeader()
	// store label and qid
	labeltemp->at(i) = label;
	if (isrankingset)
		qidtemp->at(i) = qid;

	// get feature values
	string cfeature, cvalue;
	int feature = -1;
	double value = -1.f;

	// ignore qid, if present and not isrankingset
	string qidstr("qid");
	if (not parseFeatureValue(cfeature, cvalue))
		return true;

	if (not qidstr.compare(cfeature)) // qid is present
		if (not parseFeatureValue(cfeature, cvalue))
			return true;
	do {
		// check value
		if (cvalue.empty()) {
			fprintf(
					stderr,
					"Error: invalid feature/value pair in validation/test file\n");
			return false;
		}

		// record feature
		feature = atoi(cfeature.c_str()) - 1;
		if (feature < 0 or feature > getNumFeatures()) {
			fprintf(stderr, "Error: feature index %d out of expected range\n",
					feature);
			return false;
		}
		value = (double) atof(cvalue.c_str());
		SparseFeature sf;
		sf.i_index = i;
		sf.value = value;
		rawfeatures[feature].push_back(sf);
		// TODO: faster?? consider dropping this if statement and using 
		// mine = (f >= minfeature and f <= maxfeature); features[f*mine]->value = v*mine;
	} while (parseFeatureValue(cfeature, cvalue));

	// clean up
	free(line);

	// return
	return true;
}

int InstanceData::binarySearch(int f, int i) {
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

void InstanceData::setNode(int i, int n)
{
	node[i] = n;
}

int InstanceData::getNode(int i)
{
	return node[i];
}

void InstanceData::reset() {
	// clear nodes before next tree
	for (int i = 0; i < N; i++) {
		node[i] = 0;
	}
}
bool InstanceData::parseFeatureValue(string &cfeature, string &cvalue) {
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

int InstanceData::getN() {
	return N;
}

int InstanceData::getNumFeatures() {
	return numfeatures;
}

int InstanceData::getNumQueries() {
	return numqueries;
}

double InstanceData::getFeature(int f, int i) {
	// binarySearch needs log(rawfeatures[f].size())
	int index = binarySearch(f, i);
	if (index == -1)
		return -9999999.f;

	return rawfeatures[f][index].value;
}

void InstanceData::initMetrics() {
	// if not ranking data set, return
	if (not isrankingset)
		return;

	// compute number of queries and initialize idealdcg
	numqueries = computeNumQueries(N, qid);
	idealdcg = new double[numqueries];

	// compute idealdcg for each query
	//computeIdealDCG(N, qid, label, idealdcg);
}

void InstanceData::computeMetrics(double &rmse, double &err, double &ndcg, double &rate, double &loss) {
	// uses MPI_Reduce to compute across all processors,
	// however results are only valid at root (myid==0)

	// compute rmse
	//double se = computeBoostingSE(N, label, pred);
	double se = computeMultiBoostingSE(N, K, multi_label, multi_px);
    int right_size = computeRightSize(N, K, multi_label, multi_px);
    double thisLoss = computeLogLoss(N, K, multi_label, multi_px);
     
	// compute ranking metrics
	double rawerr, rawndcg;
	int nq;
	if (isrankingset) {
		//computeBoostingRankingMetrics(N, qid, pred, label, idealdcg, rawerr, rawndcg);
		nq = getNumQueries();
	}
	// reduce sums to master
	double buffer[] = { se, (double) getN(), rawerr, rawndcg, (double) nq, (double) right_size, thisLoss};
	double recv_buffer[] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
	MPI_Reduce(&buffer, &recv_buffer, 7, MPI_DOUBLE, MPI_SUM, 0,
			MPI_COMM_WORLD);

	// compute
	rmse = sqrt(recv_buffer[0] / recv_buffer[1] * K);
    rate = recv_buffer[5] / recv_buffer[1];
    loss = recv_buffer[6];
    
	if (not isrankingset)
		return;
	err = recv_buffer[2] / recv_buffer[4];
	ndcg = recv_buffer[3] / recv_buffer[4];
}
/*
void InstanceData::updateMultiPred(int k, int i, double p) {
	multi_pred[k][i] = nanToNum(multi_pred[k][i] + p);
	double offset = 0.0;
	if (multi_pred[k][i] > 700) {
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
void InstanceData::updateMultiPred(int k, int i , double p) {
	multi_pred[k][i] = multi_pred[k][i] + p;
}

void InstanceData::updateMultiPx() {
	//double* temp = new double[N];
	for (int i = 0; i < N; i++) {
		double temp = 0.0;
		for (int k = 0; k < K; k++) {
			temp = nanToNum((temp + nanToNum(exp(multi_pred[k][i]))));
			}
		for (int k = 0; k < K; k++) {
			multi_px[k][i] = nanToNum(nanToNum(exp(multi_pred[k][i])) / temp);
		}
	}
}
/*
void InstanceData::updateMultiPx() {
	for (int i = 0; i < N; i++) {
		double temp = 0.0;
		int flag = 0;
		for (int k = 0; k < K; k++) {
			if (isinf(exp(multi_pred[k][i]))
					or multi_pred[k][i] == 1.79769313e+308) {
				multi_px[k][i] = 1.0;
				flag = 1;
				for (int j = 0; j < K; j++) {
					if (j != k) {
						multi_px[j][i] = 0.0;
					}
				}
				break;
			}
			//temp =nanToNum(temp + nanToNum(exp(multi_pred[k][i])));
			temp = temp + exp(multi_pred[k][i]);
		}
		if (flag) {
			continue;
		}
		for (int k = 0; k < K; k++) {
			//multi_px[k][i] = nanToNum(exp(multi_pred[k][i])) / temp;
			multi_px[k][i] = exp(multi_pred[k][i]) / temp;
		}
	}
}
*/
void InstanceData::predResult(string filePath) {
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
