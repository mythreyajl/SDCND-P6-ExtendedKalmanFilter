#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initializing return value
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // Return if estimations is empty
  if( estimations.size() == 0 )
  {
    std::cout << "Invalid estimations size." << std::endl;
    return rmse;
  }
  
  // Return if estimations and ground truth don't match
  if( estimations.size() != ground_truth.size() )
  {
    std::cout << "Mismatch in size of estimations and ground truth." << std::endl;
    return rmse;
  }
  
  // Accumulating errors across all measurements
  for( size_t i = 0; i < estimations.size(); i++ ) {
    auto& est = estimations[i];
    auto& gt = ground_truth[i];
    VectorXd diff = est - gt;
    VectorXd diff2 = diff.array() * diff.array();
    rmse = rmse + diff2;
  }
  // Averaging Square error to get MSE
  rmse /= estimations.size();
  
  // Getting RMSE by obtaining square root
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Initializing the result vector
  MatrixXd Hj(3, 4);
  
  // Getting references of state for ease of coding
  const double& px = x_state(0);
  const double& py = x_state(1);
  const double& vx = x_state(2);
  const double& vy = x_state(3);
  
  // Precalculate the deniominator position vector and
  // its powers for ease of representation in code
  double den = sqrt( px * px + py * py );
  double den2 = den * den;
  double den3 = den * den2;
  if( den == 0 )
      std::cout << "Error, division by 0." << std::endl;
  
  // Assigning the values according to the Jacobean calculation
  Hj(0, 0) = px / den;
  Hj(0, 1) = py / den;
  Hj(1, 0) = -py / den2;
  Hj(1, 1) = px / den2;
  Hj(2, 0) = py * ( vx * py - vy * px ) / den3;
  Hj(2, 1) = px * ( vy * px - vx * py ) / den3;
  Hj(2, 2) = px / den;
  Hj(2, 3) = py / den;
  
  return Hj;
}
