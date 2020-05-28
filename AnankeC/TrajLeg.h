#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Eigen/Geometry>
#include <iostream>
#include "Dynamics.h"
#include <pagmo/algorithm.hpp>
#include <pagmo/problem.hpp>

using namespace pagmo;

enum RegionFlags
{
	FRONT = 1,
	BACK = 2,
	PATH = 3
};

enum ObjectiveFlags
{
	LAGRANGE = 1
};

struct Constraint
{
	VecFuncType con;
	PrtFuncType dcon;
	int lcon;
	vector_double params;

	Constraint(VecFuncType con, PrtFuncType dcon, int lcon, vector_double params) :
		con(con),
		dcon(dcon),
		lcon(lcon),
		params(params) {}
};

struct Dynamics
{
	VecFuncType f;
	PrtFuncType df;
	vector_double params;

	Dynamics() {}

	Dynamics(VecFuncType f, PrtFuncType df, vector_double params) :
		f(f),
		df(df),
		params(params) {}
};

struct Objective
{
	ScaFuncType f;
	PrtFuncType df;
	ObjectiveFlags objType;
	vector_double params;

	Objective() {}

	Objective(ScaFuncType f, PrtFuncType df, ObjectiveFlags objType, vector_double params) :
		f(f),
		df(df),
		objType(objType),
		params(params) {}
};


struct TrajLeg
{
	Dynamics f;
	Objective J;
	py::list dynParams;
	std::vector<Constraint> coneqs_f;
	std::vector<Constraint> coneqs_b;
	std::vector<Constraint> coneqs_p;
	std::vector<Constraint> conins_f;
	std::vector<Constraint> conins_b;
	std::vector<Constraint> conins_p;

	double T;
	int num_nodes;
	int lenX;
	int lenU;
	int lenN;
	bool dynamicsSet;
	bool objSet;
	double Tmin;
	double Tmax;
	bool TOFset;
	vector_double bnds_min;
	vector_double bnds_max;

	TrajLeg(int n, double T){
		this->num_nodes = n;
		this->T = T;
		this->f = Dynamics();
		this->J = Objective();
	};

	void set_dynamics(VecFuncType f, PrtFuncType df, vector_double params)
	{
		this->f = Dynamics(f, df, params);
		this->dynamicsSet = true;
	}

	void set_len_X_U(int X, int U)
	{
		this->lenX = X;
		this->lenU = U;
		this->lenN = X + U;
	}

	void add_eq(VecFuncType con, PrtFuncType dcon, int lcon, RegionFlags reg, vector_double params)
	{
		switch (reg)
		{
		case FRONT:
			coneqs_f.push_back(Constraint(con, dcon, lcon, params));
			break;
		case BACK:
			coneqs_b.push_back(Constraint(con, dcon, lcon, params));
			break;
		case PATH:
			coneqs_p.push_back(Constraint(con, dcon, lcon, params));
			break;
		}
		return;
	}

	void add_ineq(VecFuncType con, PrtFuncType dcon, int lcon, RegionFlags reg, vector_double params)
	{
		switch (reg)
		{
		case FRONT:
			conins_f.push_back(Constraint(con, dcon, lcon, params));
			break;
		case BACK:
			conins_b.push_back(Constraint(con, dcon, lcon, params));
			break;
		case PATH:
			conins_p.push_back(Constraint(con, dcon, lcon, params));
			break;
		}
		return;
	}

	int getTotLength()
	{
		int totLen = 1 + this->num_nodes * (this->lenX + this->lenU);
		return totLen;
	}

	void set_TOF(double Tmin, double Tmax)
	{
		this->Tmin = Tmin;
		this->Tmax = Tmax;
		this->TOFset = true;
	}

	void set_bounds(vector_double bnds_min, vector_double bnds_max)
	{
		this->bnds_min = bnds_min;
		this->bnds_max = bnds_max;
	}

	void set_obj(ScaFuncType f, PrtFuncType df, ObjectiveFlags typ, vector_double params)
	{
		this->J = Objective(f, df, typ, params);
		this->objSet = true;
	}


	static void Build(py::module& m)
	{
		py::enum_<RegionFlags>(m, "RegionFlags")
			.value("FRONT", RegionFlags::FRONT)
			.value("BACK", RegionFlags::BACK)
			.value("PATH", RegionFlags::PATH)
			.export_values();
		py::enum_<ObjectiveFlags>(m, "ObjectiveFlags")
			.value("LAGRANGE", ObjectiveFlags::LAGRANGE)
			.export_values();

		auto obj = py::class_<TrajLeg>(m, "TrajLeg");
		obj.def(py::init<int, double>());
		obj.def("set_dynamics", &TrajLeg::set_dynamics);
		obj.def("set_len_X_U", &TrajLeg::set_len_X_U);
		obj.def("add_eq", &TrajLeg::add_eq);
		obj.def("add_ineq", &TrajLeg::add_ineq);
		obj.def("getTotLength", &TrajLeg::getTotLength);
		obj.def("set_TOF", &TrajLeg::set_TOF);
		obj.def("set_bounds", &TrajLeg::set_bounds);
		obj.def("set_obj", &TrajLeg::set_obj);
		
		obj.def_readwrite("num_nodes", &TrajLeg::num_nodes);
		obj.def_readwrite("T", &TrajLeg::T);
		obj.def_readwrite("f", &TrajLeg::f);
		obj.def_readwrite("J", &TrajLeg::J);
		obj.def_readwrite("dynamicsSet", &TrajLeg::dynamicsSet);
		obj.def_readwrite("coneqs_f", &TrajLeg::coneqs_f);
		obj.def_readwrite("coneqs_b", &TrajLeg::coneqs_b);
		obj.def_readwrite("coneqs_p", &TrajLeg::coneqs_p);
		obj.def_readwrite("conins_f", &TrajLeg::conins_f);
		obj.def_readwrite("conins_b", &TrajLeg::conins_b);
		obj.def_readwrite("conins_p", &TrajLeg::conins_p);
		obj.def_readwrite("lenX", &TrajLeg::lenX);
		obj.def_readwrite("lenU", &TrajLeg::lenU);
		obj.def_readwrite("lenN", &TrajLeg::lenN);
		obj.def_readwrite("objSet", &TrajLeg::objSet);
		obj.def_readwrite("Tmin", &TrajLeg::Tmin);
		obj.def_readwrite("Tmax", &TrajLeg::Tmax);
		obj.def_readwrite("TOFset", &TrajLeg::TOFset);
		obj.def_readwrite("bnds_min", &TrajLeg::bnds_min);
		obj.def_readwrite("bnds_max", &TrajLeg::bnds_max);

		obj.def(
			py::pickle(
				[](const TrajLeg& tl) {
					return py::make_tuple(
						tl.num_nodes, 
						tl.T,
						tl.f,
						tl.J,
						tl.dynamicsSet,
						tl.coneqs_f,
						tl.coneqs_b,
						tl.coneqs_p,
						tl.conins_f,
						tl.conins_b,
						tl.conins_p,
						tl.lenX,
						tl.lenU,
						tl.lenN,
						tl.objSet,
						tl.Tmin,
						tl.Tmax,
						tl.TOFset,
						tl.bnds_min,
						tl.bnds_max
						);
				},
				[](py::tuple t) {
					TrajLeg tl(t[0].cast<int>(), t[1].cast<double>());
					tl.num_nodes = t[0].cast<int>();
					tl.T = t[1].cast<double>();
					tl.f = t[2].cast<Dynamics>();
					tl.J = t[3].cast<Objective>();
					tl.dynamicsSet = t[4].cast<bool>();
					tl.coneqs_f = t[5].cast<std::vector<Constraint>>();
					tl.coneqs_b = t[6].cast<std::vector<Constraint>>();
					tl.coneqs_p = t[7].cast<std::vector<Constraint>>();
					tl.conins_f = t[8].cast<std::vector<Constraint>>();
					tl.conins_b = t[9].cast<std::vector<Constraint>>();
					tl.conins_p = t[10].cast<std::vector<Constraint>>();
					tl.lenX = t[11].cast<int>();
					tl.lenU = t[12].cast<int>();
					tl.lenN = t[13].cast<int>();
					tl.objSet = t[14].cast<bool>();
					tl.Tmin = t[15].cast<double>();
					tl.Tmax = t[16].cast<double>();
					tl.TOFset = t[17].cast<bool>();
					tl.bnds_min = t[18].cast<vector_double>();
					tl.bnds_max = t[19].cast<vector_double>();
					return tl;
				}
			)
		);

		return;
	}


};