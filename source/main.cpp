#include "args.h"
#include "metrics.h"
#include "loss.h"
#include "FeatureData.h"
#include "InstanceData.h"
#include "StaticTree.h"
#include "SplitsBuffer.h"
//#include "Store.h"

#include <mpi.h>
#include <ostream>
#include <fstream>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;

/*
 * MAIN: initialize MPI, read arguments
 */
time_t start;
void run(args_t a);
/**
 * TODO:
 * 3-5
 * 调试验证排序与选优
 */
int main(int argc, char* argv[]) {
	// start timer
	time(&start);

	// initialize MPI
	int myid, numprocs;
	//	int flag = 1;
	//	while (flag);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	// parse command line args
	struct args_t a;
	initialize_args(a);
	if (!get_args(argc, argv, a, myid, numprocs)) {
		// if bad args, print help and exit
		if (a.isRoot)
			print_help();
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// if good args, print parameters
	if (a.isRoot)
		//print_args(a);
	// run and exit
	//	int flag = 1;
	//	while(flag);
	run(a);
	MPI_Finalize();
	if (a.isRoot)
		//printf("OK!");
	return EXIT_SUCCESS;
}

void myslice(int myid, int numprocs, int size, int &start, int &nextstart) {
	start = (int) floor(double(myid * size) / double(numprocs));
	nextstart = (int) floor(double((myid + 1) * size) / double(numprocs));
}

/*
 * READTRAININGDATA: read and sort features from training data set
 */
FeatureData* readtrainingdata(args_t a, const char *file, int N) {
	// read and return data
	// FeatureData* data = new FeatureData(N, nf, a.computeRankingMetrics, startf, nextstartf-1);
	FeatureData* data = new FeatureData(N, a.classSize, a.numFeatures,
			a.computeRankingMetrics, a.myid, a.numProcs);

	// read, prep, and return data
	if (not data->read(file))
		exit(1);
	if (a.computeRankingMetrics)
		data->initMetrics();
	data->sort();

	return data;
}

/*
 * READTESTDATA: read a slice of a test data set
 */
InstanceData* readtestdata(args_t a, const char *file, int size, double* bincounts) {
	// determine start and stop indices for this processor
	int start, nextstart;
	myslice(a.myid, a.numProcs, size, start, nextstart);
	int N = nextstart - start;

	// read and return data
	InstanceData* data = new InstanceData(N, a.classSize, a.numFeatures,
			a.computeRankingMetrics, start, nextstart - 1);

	if (not data->read(file, size, bincounts))
		exit(1);
	if (a.computeRankingMetrics)
		data->initMetrics();
	return data;
}

/*
 * COMPUTEMETRICS: compute and print metrics
 */
void printmetricsheader(args_t a) {
	// write train metrics header
	printf("#iteration,train_rmse");
	if (a.computeRankingMetrics)
		printf(",train_err,train_ndcg");

	// write validation metrics header
	if (a.useValidSet) {
		printf(",valid_rmse");
		if (a.computeRankingMetrics)
			printf(",valid_err,valid_ndcg");
	}

	// write test metrics header
	if (a.useTestSet) {
		printf(",test_rmse");
		if (a.computeRankingMetrics)
			printf(",test_err,test_ndcg");
	}

	// print endline
	printf("\n");
}

void computemetrics(args_t a, FeatureData* train, InstanceData* valid,
		InstanceData* test, int iter) {
	// metrics variables
	double rmse, err, ndcg;

	// compute train metrics
	if (a.isRoot) {
		train->computeMetrics(rmse, err, ndcg);
		printf("iter: %d, Train rmse: %f", iter, rmse);
		if (a.computeRankingMetrics)
			printf(",%f,%f", err, ndcg);
	}

	// compute validation metrics
	if (a.useValidSet) {
		valid->computeMetrics(rmse, err, ndcg);
		if (a.isRoot) {
			printf("Valid rmse: %f", rmse);
			if (a.computeRankingMetrics)
				printf(",%f,%f", err, ndcg);
		}
	}

	// compute test metrics
	if (a.useTestSet) {
		test->computeMetrics(rmse, err, ndcg);
		if (a.isRoot) {
			printf("Test rmse: %f", rmse);
			if (a.computeRankingMetrics)
				printf(",%f,%f", err, ndcg);
		}
	}

	// print endline
	if (a.isRoot)
		printf("\n");
}

/*
 * BUILDTREE: build a regression tree in parallel
 */
void buildtree(args_t args, StaticTree* tree, FeatureData* data,
		SplitsBuffer* splits, int maxDepth, int numProcs, int numtree, int k);

/*
 * SHUFFLE: random shuffle function for Friedman subsampling
 */
static void shuffle(int *x, int n) {
	for (int i = 0; i < n; i++) {
		int j = rand() % n;
		int temp = x[i];
		x[i] = x[j];
		x[j] = temp;
	}
}

/*
 * PRINTTIME: print the elapsed time from the start to the current event
 */
static void printtime(const char* event) {
	time_t end;
	time(&end);
	printf("#time %s %f\n", event, difftime(end, start));
}

/*
 * RUN: read data, build trees, and track metrics
 */
void run(args_t a) {
	// read section of training set
	//Store* storage;
	FeatureData* train = readtrainingdata(a, a.trainFile, a.sizeTrainFile);

	// initialize splits buffer
	SplitsBuffer* splitsbuffer = new SplitsBuffer(train->getN());

	// read section of validation set
	InstanceData* valid = NULL;
	if (a.useValidSet)
		valid = readtestdata(a, a.validFile, a.sizeValidFile, train->bincounts);
	// read section of test set
	InstanceData* test = NULL;
	if (a.useTestSet)
		test = readtestdata(a, a.testFile, a.sizeTestFile, train->bincounts);
	// construct tree
	StaticTree* tree = new StaticTree(a.maxDepth, a.sizeTrainFile);
	// print sorting time
	if (a.isRoot and a.time)
		printtime("initialization");

	// print metrics header
	if (a.isRoot)
		//printmetricsheader(a);

	// construct trees

	if (a.isRoot) {
		//storage = new Store("localhost", "test.alg");
	}
	else{
		//storage = NULL;
	}
	for (int i = 0; i < a.numTrees; i++) {
		if(a.isRoot) { 
			//printf("===================================================%d\n", i);
		}
		// update residuals
		train->updateMultiPx();
		train->updateMultiResiduals();
		if (a.useValidSet)
			valid->updateMultiPx();
		if (a.useTestSet)
			test->updateMultiPx();

		//if (i % 10 == 1)
			//computemetrics(a, train, valid, test, i);

		// print tree time
		if (i % 100 == 99 and a.isRoot and a.time)
			printtime("trees");
		for (int k = 0; k < a.classSize; k++) {
			// build tree
			tree->clear();
			buildtree(a, tree, train, splitsbuffer, a.maxDepth, a.numProcs, i,
					k);

			// print tree
			if (a.isRoot) {
				//storage->insertTree(k, i, tree, a.learningRate);
				//tree->printTree(a.learningRate);
			}

			// update predictions
			tree->updateTrainingPredictions(train, k, a.learningRate);
			//TODO change updatePredictions
			if (a.useValidSet)
				tree->updatePredictions(valid, k, a.learningRate);
			if (a.useTestSet)
				tree->updatePredictions(test, k, a.learningRate);

			// compute and print metrics

		}
	}
	train->predResult();
//	if(a.useValidSet)
//		valid->predResult();

	// destroy tree
	delete tree;

	// delete datasets
	delete train;
	delete splitsbuffer;
	delete valid;
	delete test;

	// print finish time
	if (a.isRoot and a.time)
		printtime("finish");
}

/*
 * BUILD TREE: compress features, send to master, receive splits, and repeat
 */
void buildtree(args_t args, StaticTree* tree, FeatureData* data,
		SplitsBuffer* splits, int maxDepth, int numProcs, int numtree, int k) {
	// reset nodes
	data->reset();
	// apply subsampling

	// build tree in layers
	for (int l = 1; l < maxDepth; l++) {
		// find best splits on local features
		tree->findBestLocalSplits(data, k, numtree);

		// exchange local splits and determine best global splits
		tree->exchangeBestSplits();

		if (numProcs == 1) {
			// apply splits
			splits->updateSingleCore(data, tree, numtree);
		} else {
			// determine if any splitting features are stored locally
			bool localSplits = tree->containsSplittingFeature(data);

			// compute local splits buffer
			if (localSplits)
				splits->updateFromData(data, tree);
			else
				splits->clear();

			// exchange to find global splits buffer
			if (l == 1) {
				int feature;
				double split;
				tree->getSplit(0, feature, split);
				//cout << "gfeature: " << feature << "gsplit: " << split << endl;

				int root = data->whoHasFeature(feature);
				splits->broadcast(root);
			} else
				splits->exchange();
			// apply splits
			splits->applyToData(data);
		}
		// start next layer
		tree->startNextLayer();
	}
}
