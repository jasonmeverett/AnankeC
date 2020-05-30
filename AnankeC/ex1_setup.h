#pragma once

#include "Structs.h"

namespace py = pybind11;


struct Ex1
{
	inline static Eigen::VectorXd f(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 2, 1> dX;
		dX[0] = X[1];
		dX[1] = U[0];
		return dX;
	};

	inline static std::vector<Eigen::MatrixXd> df(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 2, 2> dfdX = Eigen::Matrix<double, 2, 2>::Zero();
		Eigen::Matrix<double, 2, 1> dfdU = Eigen::Matrix<double, 2, 1>::Zero();
		Eigen::Matrix<double, 2, 1> dfdT = Eigen::Matrix<double, 2, 1>::Zero();
		dfdX(0, 1) = 1.0;
		dfdU(1, 0) = 1.0;
		return {dfdX, dfdU, dfdT};
	};

	inline static double Jctrl(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		return U[0] * U[0];
	};

	inline static std::vector<Eigen::MatrixXd> dJctrl(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 1, 2> dJdX = Eigen::Matrix<double, 1, 2>::Zero();
		Eigen::Matrix<double, 1, 1> dJdU = Eigen::Matrix<double, 1, 1>::Zero();
		dJdU(0, 0) = 2.0 * U[0];
		return {dJdX, dJdU};
	};

	inline static Eigen::VectorXd g1(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 2, 1> g(2);
		g[0] = X[0];
		g[1] = X[1];
		return g;
	};

	inline static std::vector<Eigen::MatrixXd> dg1(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::MatrixXd dgdX = Eigen::MatrixXd::Zero(2, 2);
		Eigen::MatrixXd dgdU = Eigen::MatrixXd::Zero(2, 1);
		Eigen::MatrixXd dgdT = Eigen::MatrixXd::Zero(2, 1);
		dgdX(0, 0) = 1.0;
		dgdX(1, 1) = 1.0;
		return { dgdX, dgdU, dgdT };
	};

	inline static Eigen::VectorXd g2(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 2, 1> g(2);
		g[0] = X[0] - 1.0;
		g[1] = X[1];
		return g;
	};

	inline static std::vector<Eigen::MatrixXd> dg2(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::MatrixXd dgdX = Eigen::MatrixXd::Zero(2, 2);
		Eigen::MatrixXd dgdU = Eigen::MatrixXd::Zero(2, 1);
		Eigen::MatrixXd dgdT = Eigen::MatrixXd::Zero(2, 1);
		dgdX(0, 0) = 1.0;
		dgdX(1, 1) = 1.0;
		return { dgdX, dgdU, dgdT };
	};

	inline static Eigen::VectorXd g3(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::Matrix<double, 1, 1> g(1);
		double ulim = params[0];
		g[0] = sqrt(U[0] * U[0]) - ulim;
		return g;
	};

	inline static std::vector<Eigen::MatrixXd> dg3(Eigen::VectorXd X, Eigen::VectorXd U, double T, vector_double params) {
		Eigen::MatrixXd dgdX = Eigen::MatrixXd::Zero(1, 2);
		Eigen::MatrixXd dgdU = Eigen::MatrixXd::Zero(1, 1);
		Eigen::MatrixXd dgdT = Eigen::MatrixXd::Zero(1, 1);
		dgdU(0, 0) = U[0] / sqrt(U[0] * U[0]);
		return { dgdX, dgdU, dgdT };
	};

	static void Build(py::module& m)
	{
		auto obj = py::class_<Ex1>(m, "ex1");
		obj.def_static("f", &Ex1::f);
		obj.def_static("df", &Ex1::df);
		obj.def_static("Jctrl", &Ex1::Jctrl);
		obj.def_static("dJctrl", &Ex1::dJctrl);
		obj.def_static("g1", &Ex1::g1);
		obj.def_static("g2", &Ex1::g2);
		obj.def_static("g3", &Ex1::g3);
		obj.def_static("dg1", &Ex1::dg1);
		obj.def_static("dg2", &Ex1::dg2);
		obj.def_static("dg3", &Ex1::dg3);
	}
};
