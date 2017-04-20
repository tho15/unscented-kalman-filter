#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = false;

	// state dimension is 5 (px, py, v, yaw, yaw_rate]
	n_x_   = 5;
	n_aug_ = 7;  // n_x_ + 2
	
	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 0.8;  // 3 m/s^2 for bicycle
	
	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	/**
	   Complete the initialization. See ukf.h for other member properties.
	*/
	is_initialized_ = false;
	
	P_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1;
	
	// initialize weights
	weights_ = VectorXd(15);

	double lambda = 3 - n_aug_;

	weights_(0) = (double) lambda/(lambda+n_aug_);
	for (int i=1; i < 2*n_aug_ +1; i++) {
		weights_(i) = (double) 0.5 / (n_aug_ +lambda);
	}
	
	NIS_radar_ = 0;
	NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * generate sigma points
 */
void UKF::GenerateSigmaPoints(MatrixXd &Xsig)
{
	double lambda = 3 - n_x_;
	
	//calculate square root of P
	MatrixXd A = P_.llt().matrixL();
	
	Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
	//set sigma points as columns of matrix Xsig
	Xsig.col(0) = x_;
  
	// multiple square root of P with square root of 3
	A = std::sqrt(lambda+n_x_)*A;
	for (int i = 1; i < 2*n_x_ +1; i++) {
		if(i > n_x_) {
			Xsig.col(i) = x_ - A.col(i-n_x_ -1);
		} else {
			Xsig.col(i) = x_ + A.col(i-1);
		}
	}
}

inline void UKF::NormalizeAngle(double &angle)
{
	while (angle > M_PI) angle-=2.*M_PI;
	while (angle <-M_PI) angle+=2.*M_PI;
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd& Xsig_aug)
{
	// augmented mean vector
	VectorXd x_aug = VectorXd(7);
	// augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);
	int n_aug = n_x_ +2;
	
	double lambda = 3 - n_aug;

	// sigma point matrix
	Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
	
	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;
	
	// create augmented covariance matrix
	//std::cout << "P_ is " << std::endl << P_ << std::endl;
	P_aug.fill(0.0);
	P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
	P_aug(5, 5) = std_a_*std_a_;
	P_aug(6, 6) = std_yawdd_*std_yawdd_;
	//std::cout << "P_aug is " << std:: endl << P_aug << std::endl;
  
	// create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	if (P_aug.llt().info() == Eigen::NumericalIssue) {
		std::cout << "llt failed! we have numerical issue" << std::endl;
		throw std::range_error("llt numerical issue!");
	}
	A = std::sqrt(lambda+n_aug)*A;
	//std::cout << "A is: " << std::endl << A << std::endl;
	
	// create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for(int i = 1; i <= n_aug; i++) {
		Xsig_aug.col(i) = x_aug + A.col(i-1);
		Xsig_aug.col(i+n_aug) = x_aug - A.col(i-1);
	}
}

void UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t)
{
	int n_aug = n_x_ +2;
	
	Xsig_pred_ = MatrixXd(n_x_, 2*n_aug+1);
	
	//predict sigma points
	for (int i = 0; i< 2*n_aug+1; i++) {
		//extract values for better readability
		double p_x  = Xsig_aug(0,i);
		double p_y  = Xsig_aug(1,i);
		double v    = Xsig_aug(2,i);
		double yaw  = Xsig_aug(3,i);
		double yawd = Xsig_aug(4,i);
		
		double nu_a = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.0001) {
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}
		
		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;
		
		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		// normalize yaw
		NormalizeAngle(yaw_p);

		//write predicted sigma point into right column
		Xsig_pred_(0,i) = px_p;
		Xsig_pred_(1,i) = py_p;
		Xsig_pred_(2,i) = v_p;
		Xsig_pred_(3,i) = yaw_p;
		Xsig_pred_(4,i) = yawd_p;
	}
}

void UKF::InitMeasurement(MeasurementPackage meas_package)
{
	time_us_ = meas_package.timestamp_;
	  
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		/**
		 Convert radar from polar to cartesian coordinates and initialize state.
		 */
		double px = meas_package.raw_measurements_[0]*std::cos(meas_package.raw_measurements_[1]);
		double py = meas_package.raw_measurements_[0]*std::sin(meas_package.raw_measurements_[1]);
		
		x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER & use_laser_) {
		/**
		 Initialize state.
		*/
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	}

    // done initializing, no need to predict or update
    is_initialized_ = true;
	last_meas = meas_package;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{	
	// initialize state vector if not yet
	if(!is_initialized_) {
		InitMeasurement(meas_package);
		return;
	}

	/* prediction */
	double delta_t = (double)(meas_package.timestamp_ - time_us_)/1000000.0;

	try {
		if(delta_t > 0.2) {
			const double dt = delta_t/2.0;
			Prediction(dt);
			delta_t -= dt;
		}
		
		Prediction(delta_t);
	} catch(std::range_error e) {
		P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;
		// skip prediction step?
		InitMeasurement(last_meas);
		return;  // skip this measurement
	}
	
	/* then update base on measurement */
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		UpdateLidar(meas_package);
	}
	last_meas = meas_package;
	time_us_  = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
	int n_aug = n_x_ +2;
	
	// calculate predicted sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug, 2*n_aug +1);
	GenerateAugmentedSigmaPoints(Xsig_aug);
	PredictSigmaPoints(Xsig_aug, delta_t);
		    
	//predict state mean
	x_ = Xsig_pred_*weights_;

	P_.fill(0.0);
	for(int i = 1; i < 2*n_aug+1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
		//VectorXd x_diff = Xsig_pred_.col(i) - x_;
		NormalizeAngle(x_diff(3));
		P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
	}
 }

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  UKFLaserMeasurement m_laser(*this);
  m_laser.UpdateState(meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
	UKFRadarMeasurement m_radar(*this);
	m_radar.UpdateState(meas_package);
}


void UKFMeasurement::UpdateState(const MeasurementPackage &z)
{
	int n_aug = ukf_.n_aug_;

	// sigma points in measurement space.
	if(!PredictMeasurement(z)) {
		std::cout << "Update skip, measurement prediction fails" << std::endl;
		return;
	}
	
	MatrixXd Tc = MatrixXd(ukf_.n_x_, n_z_);
	
	Tc.fill(0.0);
	//calculate cross correlation matrix
	for(int i = 1; i < 2 * n_aug + 1; i++) {
		VectorXd Xsig_diff = ukf_.Xsig_pred_.col(i) - ukf_.Xsig_pred_.col(0);
		//VectorXd Xsig_diff = ukf_.Xsig_pred_.col(i) - ukf_.x_;
		ukf_.NormalizeAngle(Xsig_diff(3));
		
		VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
		ukf_.NormalizeAngle(z_diff(1));
		
		Tc = Tc + ukf_.weights_(i)*Xsig_diff*z_diff.transpose();
	}
	
	//calculate Kalman gain K;
	MatrixXd K(ukf_.n_x_, n_z_);
	K = Tc*S_.inverse();
	
	//update state mean and covariance matrix
	ukf_.x_ += K*(z_ - z_pred_);
	ukf_.P_ -= K*S_*K.transpose();	
}

/*
 * this function calculates predicted mean measurement and covariance in
 * measurement space for nonlinear transform.
 */
bool UKFRadarMeasurement::PredictMeasurement(const MeasurementPackage &z)
{	
	int n_aug = ukf_.n_aug_;

	if(z.sensor_type_ != MeasurementPackage::RADAR) {
		cout << "UKFRadarMeasurement::Init error invalid sensor_type! " 
			 << z.sensor_type_ << endl;
		return false;
	}
	z_ = z.raw_measurements_;
		
	Zsig_pred_ = MatrixXd(n_z_, 2 * n_aug + 1);
	z_pred_ = VectorXd(n_z_);
	
	MatrixXd S = MatrixXd(n_z_, n_z_);
	
	//transform sigma points into radar measurement space
	for(int i = 0; i < 2*n_aug +1; i++) {
		double px  = ukf_.Xsig_pred_(0, i);
		double py  = ukf_.Xsig_pred_(1, i);
		double v   = ukf_.Xsig_pred_(2, i);
		double phi = ukf_.Xsig_pred_(3, i);
		
		if(px < 0.000001) px = 0.00001;
		Zsig_pred_(0, i) = std::sqrt(px*px + py*py);
		if(std::fabs(Zsig_pred_(0, i)) < 0.00001) {
			std::cout << "rho is zero, invalid px/py!" << std::endl;
			return false;
		}
		Zsig_pred_(1, i) = std::atan2(py, px);
		Zsig_pred_(2, i) = (px*std::cos(phi)*v + py*std::sin(phi)*v)/Zsig_pred_(0, i);
	}
	
	//calculate mean predicted measurement
	z_pred_ = Zsig_pred_*ukf_.weights_;
	// normalize phi
	ukf_.NormalizeAngle(z_pred_(1));
	
	//calculate measurement covariance matrix S
	S_ = MatrixXd(n_z_, n_z_);
	S_.fill(0.0);
	for(int i = 1; i < 2*n_aug +1; i++) {
		VectorXd z_diff = Zsig_pred_.col(i) - Zsig_pred_.col(0);
		//VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
		// angle normalization
		ukf_.NormalizeAngle(z_diff(1));
		S_ = S_ + ukf_.weights_(i)*z_diff*z_diff.transpose();
	}
  	
	// add R noise covariance
	S_(0, 0) += ukf_.std_radr_*ukf_.std_radr_;
	S_(1, 1) += ukf_.std_radphi_*ukf_.std_radphi_;
	S_(2, 2) += ukf_.std_radrd_*ukf_.std_radrd_;
	
	VectorXd zd = z_ - z_pred_;
	
	ukf_.NIS_radar_ = zd.transpose()*S_.inverse()*zd;
	
	return true;
}

UKFLaserMeasurement::UKFLaserMeasurement(UKF &ukf): UKFMeasurement(ukf)
{
	n_z_ = 2;
	
	H_ = MatrixXd(n_z_, ukf_.n_x_);
	H_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0;
	
	R_ = MatrixXd(n_z_, n_z_);
	R_ << ukf_.std_laspx_*ukf_.std_laspx_, 0,
		0, ukf_.std_laspy_*ukf_.std_laspy_;
}

/* use linear prediction for laser state update */
void UKFLaserMeasurement::UpdateState(const MeasurementPackage &z)
{
	VectorXd y  = z.raw_measurements_ - H_*ukf_.x_;
	MatrixXd Ht = H_.transpose();
	
	S_ = H_*ukf_.P_*Ht + R_;
	MatrixXd K  = ukf_.P_*Ht*S_.inverse();
	MatrixXd I  = MatrixXd::Identity(ukf_.n_x_, ukf_.n_x_);
  
	// measurement update
	ukf_.x_ = ukf_.x_ + K*y;
	ukf_.P_ = (I - K*H_)*ukf_.P_;
	
	ukf_.NIS_laser_ = y.transpose()*S_.inverse()*y;
}


