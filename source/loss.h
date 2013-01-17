#ifndef LOSS_H
#define LOSS_H

void computegradient(int N, double *label, double *pred, int *qid, double *residual) {
    for (int i=0; i<N; i++)
        residual[i] = label[i] - pred[i]; // gradient of squared-loss w.r.t. each instance
}
#endif
