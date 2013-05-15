#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <vector>
#include <math.h>
#include <map>
#include <queue>

using namespace std;

/*
 *  conditional exp computing
 */
static double conditionalExp(double n) {
	if (n > 709)
	{
		printf("hit\n");
		return 1.79769313e+307;
	}
	else {
		if(exp(n) > 1.79769313e+307) {
			return 1.79769313e+307;
		}	
		else {
			return exp(n);
		}
	}

}
static double nanToNum(double n) {
	if(!isfinite(n)){
		return 1.79769313e+308;
	}
	else {
		return n;
	}
}

/*
 * compute SE
 */
static double computeBoostingSE(int N, double *label, double *pred) {
	// computes SE between label and pred values
	double SE = 0.;
	for (int i=0; i<N; i++)
		SE += pow(label[i] - pred[i], 2.0);
	return SE;
}
/*
 * compute MultiBoostingSE
 */
static double computeMultiBoostingSE(int N, int K, double **multi_label, double **multi_px) {
	double SE = 0.;
	for (int i=0; i<N; i++) {
		for (int k=0; k<K; k++) {
			SE += pow(multi_label[k][i] - multi_px[k][i], 2.0);
		}
	}
	return SE;
}
static double computeLogLoss(int N, int K, double **multi_label, double **multi_px) {
    double loss;
    for (int i=0; i<N; i++) {
        for (int k = 0; k < K; k++) {
            loss += multi_label[k][i] * log(multi_px[k][i]); 
        }
    }
    return -1 * loss;
}

static int computeRightSize(int N, int K, double **multi_label, double **multi_px) {
    int right_size = 0;
    for (int i=0; i<N; i++) {
        double max_value, max_index;
        max_value = max_index = 0.0;
        for (int k=0; k<K; k++){
            if(multi_px[k][i] >= max_value) {
                max_value = multi_px[k][i];
                max_index = k;
            }
        }
        if (multi_label[(int)max_index][i] == 1.0) {
            right_size ++;
        }
    }
    return right_size;
}

/*
 * compute number of queries (assumes, per data format standard, that instances are ordered by qid)
 */
static int computeNumQueries(int N, int *qid) {
	// initialize count
	int nq = 1;
	int prevqid = qid[0];

	// count qids
	for (int i=1; i<N; i++) {
		nq += (qid[i] != prevqid);
		prevqid = qid[i];
	}

	return nq;
}

/*
 * compute ideal DCG
 */
struct Document {
	double pred;
	double label;
};

class CompareDocuments {
	public: bool operator() (Document d1, Document d2) { return (d1.pred < d2.pred); }
};

static double R(double y) {
	return (pow(2.0,y) - 1.0) / 16.0;
}

static void computeIdealDCG(int N, int *qid, double *label, double *idealdcg) {
	priority_queue<
		Document,vector<Document>,CompareDocuments> rank;
	int i = 0;
	int nq = 0;

	// iterate over queries
	while(i<N) {
		// get query id
		int currqid = qid[i];
		nq++;

		// add documents for that query to priority queue
		while (i<N and qid[i] == currqid) {
			Document doc;
			doc.pred = label[i];
			doc.label = label[i];
			rank.push(doc);
			i++;
		}

		// compute ideal ndcg
		double dcg = 0.;
		for (int j=1; !rank.empty() and j<=10; j++) {
			// get label of document
			Document doc = rank.top();
			rank.pop();

			// add to dcg, if in top 10
			dcg += (pow(2.0,doc.label) - 1.0) / (log2(1.0+j));
		}

		// delete remaining docs
		while (!rank.empty()) {
			rank.pop();
		}

		// set ideal dcg for this query
		idealdcg[nq-1] = dcg;
	}
}

/*
 * compute rawERR and rawNDCG
 */
static void computeBoostingRankingMetrics(int N, int *qid, double *pred, double *label, double* idealdcg, double &rawerr, double &rawndcg) {
	priority_queue<
		Document,vector<Document>,CompareDocuments> rank;
	rawerr = 0.0;
	rawndcg = 0.0;

	int i = 0;
	int nq = 0;
	// iterate over queries
	while (i<N) {
		// get query id
		int currqid = qid[i];
		nq++;

		// add documents for that query to priority queue
		while (i<N and qid[i] == currqid) {
			Document doc;
			doc.pred = pred[i];
			doc.label = label[i];
			rank.push(doc);
			i++;
		}

		// compute err and ndcg
		double p = 1.0;
		double dcg = 0.0;
		for (int j=1; !rank.empty(); j++) {
			// get label of document
			Document doc = rank.top();
			rank.pop();

			// add to err
			rawerr += 1.0/j * R(doc.label) * p;
			p *= (1.0 - R(doc.label));

			// add to dcg, if in top 10
			if (j <= 10) dcg += (pow(2.0,doc.label) - 1.0) / (log2(1.0+j));
		}

		// add to ndcg
		if (idealdcg[nq-1] > 0)
			rawndcg += dcg / idealdcg[nq-1];
		else rawndcg += 1.0;
	}
}

#endif
