#include "kalman_filter.h"
#include <iostream>

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
    std::cout << "x_ b4:" << std::endl << x_ << std::endl;
    x_ = F_ * x_;
    std::cout << "x_ after:" << std::endl << x_ << std::endl;
    P_ = F_ * P_ * F_.transpose() + Q_;
    std::cout << "P_ predicted:" << std::endl << P_ << std::endl;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * S.inverse();
    std::cout << "K measured LiDAR:" << std::endl << P_ << std::endl;

    x_ = x_ + K * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
    std::cout << "P_ measured LiDAR:" << std::endl << P_ << std::endl;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    auto& Hj = H_;
    VectorXd z_pred = Eigen::VectorXd(3);
    z_pred(0) = sqrt(x_(0)*x_(0) + x_(1) * x_(1));
    z_pred(1) = atan2(x_(1), x_(0));
    z_pred(2) = (x_(0)*x_(2) + x_(1) * x_(3))/z_pred(0);
    VectorXd y = z - z_pred;
    MatrixXd Ht = Hj.transpose();
    MatrixXd S = Hj * P_ * Ht + R_;
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * S.inverse();
    std::cout << "K measured RADAR:" << std::endl << P_ << std::endl;


    x_ = x_ + K * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * Hj) * P_;
    std::cout << "P_ measured RADAR:" << std::endl << P_ << std::endl;
}
