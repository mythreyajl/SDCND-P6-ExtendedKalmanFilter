#define _USE_MATH_DEFINES

#include "kalman_filter.h"
#include <iostream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * S.inverse();

    x_ = x_ + K * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    auto& Hj = H_;
    
    // Measurement vs. Prediction
    VectorXd z_pred = Eigen::VectorXd(3);
    z_pred(0) = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    z_pred(1) = atan2(x_(1), x_(0));
    z_pred(2) = (x_(0) * x_(2) + x_(1) * x_(3))/z_pred(0);
    VectorXd y = z - z_pred;
    if(y(1) > M_PI) {
        while(y(1) > M_PI)
            y(1) -= 2*M_PI;
    } else if(y(1) < -M_PI) {
        while(y(1) < -M_PI)
            y(1) += 2*M_PI;
    }
    
    MatrixXd Ht = Hj.transpose();
    MatrixXd S = Hj * P_ * Ht + R_;
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * S.inverse();

    x_ = x_ + K * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * Hj) * P_;
}
