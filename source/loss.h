#ifndef LOSS_H
#define LOSS_H

void computegradient(int N, double *label, double *pred, int *qid, double *residual) {
	for (int i=0; i<N; i++)
		residual[i] = label[i] - pred[i]; // gradient of squared-loss w.r.t. each instance
}

void computeMultiGradient(int N, int K, double **multi_label, double **multi_px, int *qid, double **multi_residual) {
	for (int i=0; i<N; i++) {
		for (int k=0; k<K; k++) {
			multi_residual[k][i] = multi_label[k][i] - multi_px[k][i];//grandient of deviance-loss for each class and instance
		}
	}
}       
#endif
