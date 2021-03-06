#pragma once

#include "Structs.h"
#include "TrajLeg.h"

using namespace std::chrono;
using namespace pagmo;

struct LegLink
{
	int l1;
	int l2;
	LnkFuncType lfun;
	LprFuncType dlfun;
	int length;
	vector_double params;

	LegLink(int l1, int l2, LnkFuncType lfun, LprFuncType dlfun, int length, vector_double params) :
		l1(l1),
		l2(l2),
		lfun(lfun),
		dlfun(dlfun),
		length(length),
		params(params) {}
};

struct Ananke_Config
{
	std::vector<TrajLeg> TrajLegs;
	std::vector<LegLink> LegLinks;
	double maxTOF;
	double minTOF; 
	int idxLegObj;
	bool use_estimate_grad = false;
	double est_grad_dt = 1e-8;
	int cur_nec;
	int cur_nic;
	std::pair<vector_double, vector_double> cur_bounds;
	sparsity_pattern sparse_pattern;

	void add_leg_link(int l1, int l2, LnkFuncType lfun, LprFuncType dlfun, int length, vector_double params)
	{
		this->LegLinks.push_back(LegLink(l1, l2, lfun, dlfun, length, params));
	}

	void set_TOF(double minTOF, double maxTOF) 
	{
		this->maxTOF = maxTOF;
		this->minTOF = minTOF;
	}

	std::vector<Eigen::MatrixXd> get_array_data(Eigen::VectorXd x)
	{
		std::vector<Eigen::MatrixXd> out;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			double T0 = 0.0;
			for (int jj = 0; jj < ii; jj++)
			{
				T0 += x[this->get_dvi_T(jj)];
			}
			int idT = this->get_dvi_T(ii);
			double dt = x[idT] / static_cast<double>(TL.num_nodes-1);
			
			Eigen::MatrixXd outLeg = Eigen::MatrixXd::Zero(TL.num_nodes, 1 + TL.lenN);
			for (int jj = 0; jj < TL.num_nodes; jj++)
			{
				int id0 = this->get_dvi_N(ii, jj);
				double Tk = T0 + dt * static_cast<double>(jj);
				Eigen::VectorXd XU = x.segment(id0, TL.lenN);
				outLeg(jj, 0) = Tk;
				outLeg.block(jj, 1, 1, TL.lenN) = XU.transpose();
			}
			out.push_back(outLeg);
		}
		return out;
	}

	vector_double::size_type get_nec() { return this->cur_nec; }
	vector_double::size_type get_nic() { return this->cur_nic; }

	void set_nec() 
	{
		int num_defects = 0;
		int num_con_user = 0;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			num_defects += TL.lenX * (TL.num_nodes - 1);

			for (int jj = 0; jj < TL.coneqs_f.size(); jj++)
			{
				num_con_user += TL.coneqs_f[jj].lcon;
			}
			for (int jj = 0; jj < TL.coneqs_b.size(); jj++)
			{
				num_con_user += TL.coneqs_b[jj].lcon;
			}
			for (int jj = 0; jj < TL.coneqs_p.size(); jj++)
			{
				num_con_user += TL.coneqs_p[jj].lcon * TL.num_nodes;
			}
		}

		for (int ii = 0; ii < this->LegLinks.size(); ii++)
		{
			num_con_user += this->LegLinks[ii].length;
		}

		this->cur_nec = num_defects + num_con_user;
		return;
	}

	void set_nic() 
	{
		int num_con_user = 0;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];

			for (int jj = 0; jj < TL.conins_f.size(); jj++)
			{
				num_con_user += TL.conins_f[ii].lcon;
			}
			for (int jj = 0; jj < TL.conins_b.size(); jj++)
			{
				num_con_user += TL.conins_b[ii].lcon;
			}
			for (int jj = 0; jj < TL.conins_p.size(); jj++)
			{
				num_con_user += TL.conins_p[ii].lcon * TL.num_nodes;
			}

			if (TL.Tmin > 0.0)
			{
				num_con_user += 2;
			}
		}

		if (this->maxTOF > 0.0)
		{
			num_con_user += 2;
		}

		this->cur_nic = num_con_user;
		return;
	}

	std::pair<vector_double, vector_double> get_bounds() { return this->cur_bounds; }

	void set_bounds()
	{
		vector_double LB;
		vector_double UB;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			LB.push_back(-10000.0);
			UB.push_back(+10000.0);
			for (int jj = 0; jj < TL.num_nodes; jj++)
			{
				for (int kk = 0; kk < TL.bnds_min.size(); kk++)
				{
					LB.push_back(TL.bnds_min[kk]);
					UB.push_back(TL.bnds_max[kk]);
				}
			}
		}
		this->cur_bounds = std::make_pair(LB, UB);
		return;
	}


	inline double calc_J(Eigen::VectorXd x) 
	{
		TrajLeg TL = this->TrajLegs[this->idxLegObj];
		double T0 = 0.0;
		for (int ii = 0; ii < this->idxLegObj; ii++)
		{
			T0 += x[this->get_dvi_T(ii)];
		}
		int idT = this->get_dvi_T(this->idxLegObj);
		double dt = x[idT] / static_cast<double>(TL.num_nodes-1);

		double J = 0.0;
		if (TL.J.objType == ObjectiveFlags::LAGRANGE)
		{
			J = 0.0;
			Eigen::VectorXd Xk(TL.lenX);
			Eigen::VectorXd Uk(TL.lenU);
			Eigen::VectorXd Xkp1(TL.lenX);
			Eigen::VectorXd Ukp1(TL.lenU);
			int id0_Xk = this->get_dvi_N(this->idxLegObj, 0);
			for (int ii = 0; ii < TL.num_nodes - 1; ii++)
			{
				double Tk = T0 + dt * static_cast<double>(ii);
				double Tkp1 = T0 + dt * static_cast<double>(ii + 1);
				int id0_Uk = id0_Xk + TL.lenX;
				int id0_Xkp1 = id0_Uk + TL.lenU;
				int id0_Ukp1 = id0_Xkp1 + TL.lenX;
				Xk = x.segment(id0_Xk, TL.lenX);
				Uk = x.segment(id0_Uk, TL.lenU);
				Xkp1 = x.segment(id0_Xkp1, TL.lenX);
				Ukp1 = x.segment(id0_Ukp1, TL.lenU);
				vector_double params = TL.J.params;
				double J1 = TL.J.f(Xk, Uk, Tk, params);
				double J2 = TL.J.f(Xkp1, Ukp1, Tkp1, params);
				J += 0.5 * (J1 + J2) * dt;
				id0_Xk += TL.lenN;
			}
		}
		return J;
	}

	inline vector_double fitness(vector_double dv)
	{

		Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> x(dv.data(), dv.size());

		int num_con_user = 0;
		for (int ii = 0; ii < this->LegLinks.size(); ii++)
		{
			num_con_user += this->LegLinks[ii].length;
		}

		Eigen::VectorXd OBJVAL = Eigen::VectorXd::Zero(1);
		Eigen::VectorXd CONEQ = Eigen::VectorXd::Zero(this->get_nec() - num_con_user);
		Eigen::VectorXd CONIN = Eigen::VectorXd::Zero(this->get_nic());
		Eigen::VectorXd CONLK = Eigen::VectorXd::Zero(num_con_user);
		int idxEQ = 0;
		int idxLK = 0;
		int idxIN = 0;

		// Calculate objective.
		double J = this->calc_J(x);
		OBJVAL(0) = J;

		// Constraints
		double tofTOT = 0.0;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			double T0 = 0.0;
			for (int jj = 0; jj < ii; jj++)
			{
				T0 += x[this->get_dvi_T(jj)];
			}
			int idT = this->get_dvi_T(ii);
			double dt = x[idT] / static_cast<double>(TL.num_nodes-1);
			tofTOT += x[idT];

			// Equality constraints
			for (int jj = 0; jj < TL.coneqs_f.size(); jj++)
			{
				Constraint coneq = TL.coneqs_f[jj];
				int id0 = this->get_dvi_N(ii, 0);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0;
				vector_double params = coneq.params;
				CONEQ.segment(idxEQ, coneq.lcon) = coneq.con(X, U, T, params);
				idxEQ += coneq.lcon;
			}
			for (int jj = 0; jj < TL.coneqs_b.size(); jj++)
			{
				Constraint coneq = TL.coneqs_b[jj];
				int id0 = this->get_dvi_N(ii, -1);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0 + x[this->get_dvi_T(ii)];
				vector_double params = coneq.params;
				CONEQ.segment(idxEQ, coneq.lcon) = coneq.con(X, U, T, params);
				idxEQ += coneq.lcon;
			}
			for (int jj = 0; jj < TL.coneqs_p.size(); jj++)
			{
				Constraint coneq = TL.coneqs_p[jj];
				vector_double params = coneq.params;
				Eigen::VectorXd X(TL.lenX);
				Eigen::VectorXd U(TL.lenU);
				int id0 = this->get_dvi_N(ii, 0);
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					X = x.segment(id0, TL.lenX);
					U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + dt * static_cast<double>(kk);
					CONEQ.segment(idxEQ, coneq.lcon) = coneq.con(X, U, T, params);
					idxEQ += coneq.lcon;
					id0 += TL.lenN;
				}
			}

			// Collocation constraints
			Eigen::VectorXd Xk(TL.lenX);
			Eigen::VectorXd Uk(TL.lenU);
			Eigen::VectorXd Xkp1(TL.lenX);
			Eigen::VectorXd Ukp1(TL.lenU);
			Eigen::VectorXd fk(TL.lenX);
			Eigen::VectorXd fkp1(TL.lenX);
			Eigen::VectorXd Uc(TL.lenU);
			Eigen::VectorXd Xc(TL.lenX);
			Eigen::VectorXd fc(TL.lenX);
			Eigen::VectorXd constr_p(TL.lenX);
			vector_double params = TL.f.params;
			int id0_Xk = this->get_dvi_N(ii, 0);
			for (int jj = 0; jj < TL.num_nodes-1; jj++)
			{
				double Tk = T0 + dt * static_cast<double>(jj);
				double Tkp1 = T0 + dt * static_cast<double>(jj + 1);
				int id0_Uk = id0_Xk + TL.lenX;
				int id0_Xkp1 = id0_Uk + TL.lenU;
				int id0_Ukp1 = id0_Xkp1 + TL.lenX;
				Xk = x.segment(id0_Xk, TL.lenX);
				Uk = x.segment(id0_Uk, TL.lenU);
				Xkp1 = x.segment(id0_Xkp1, TL.lenX);
				Ukp1 = x.segment(id0_Ukp1, TL.lenU);
				fk = TL.f.f(Xk, Uk, Tk, params);
				fkp1 = TL.f.f(Xkp1, Ukp1, Tkp1, params);
				Uc = 0.5 * (Uk + Ukp1);
				Xc = 0.5 * (Xk + Xkp1) + dt / 8.0 * (fk - fkp1);
				double Tc = 0.5 * (Tk + Tkp1);
				fc = TL.f.f(Xc, Uc, Tc, params);
				constr_p = Xk - Xkp1 + dt / 6.0 * (fk + 4.0 * fc + fkp1);
				CONEQ.segment(idxEQ, TL.lenX) = constr_p;
				idxEQ += TL.lenX;
				id0_Xk += TL.lenN;
			}

			// Inequality constraints
			for (int jj = 0; jj < TL.conins_f.size(); jj++)
			{
				Constraint conin = TL.conins_f[jj];
				int id0 = this->get_dvi_N(ii, 0);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0;
				vector_double params = conin.params;
				CONIN.segment(idxIN, conin.lcon) = conin.con(X, U, T, params);
				idxIN += conin.lcon;
			}
			for (int jj = 0; jj < TL.conins_b.size(); jj++)
			{
				Constraint conin = TL.conins_b[jj];
				int id0 = this->get_dvi_N(ii, -1);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0 + x[this->get_dvi_T(ii)];
				vector_double params = conin.params;
				CONIN.segment(idxIN, conin.lcon) = conin.con(X, U, T, params);
				idxIN += conin.lcon;
			}
			for (int jj = 0; jj < TL.conins_p.size(); jj++)
			{
				Constraint conin = TL.conins_p[jj];
				vector_double params = conin.params;
				Eigen::VectorXd X(TL.lenX);
				Eigen::VectorXd U(TL.lenU);
				int id0 = this->get_dvi_N(ii, 0);
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					X = x.segment(id0, TL.lenX);
					U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + dt * static_cast<double>(kk);
					CONIN.segment(idxIN, conin.lcon) = conin.con(X, U, T, params);
					idxIN += conin.lcon;
					id0 += TL.lenN;
				}
			}

			// TOF constraints for each leg
			CONIN(idxIN) = TL.Tmin - x[idT];
			idxIN++;
			CONIN(idxIN) = x[idT] - TL.Tmax;
			idxIN++;
		}

		// Linking constraints
		for (int ii = 0; ii < this->LegLinks.size(); ii++)
		{
			LegLink LL = this->LegLinks[ii];
			int li0 = LL.l1;
			int li1 = LL.l2;
			vector_double params = LL.params;
			int id0_Xk = this->get_dvi_N(li0, -1);
			int idf_Xk = id0_Xk + this->TrajLegs[li0].lenX;
			int id0_Uk = idf_Xk;
			int idf_Uk = id0_Uk + this->TrajLegs[li0].lenU;
			int id0_Xkp1 = this->get_dvi_N(li1, 0);
			int idf_Xkp1 = id0_Xkp1 + this->TrajLegs[li1].lenX;
			int id0_Ukp1 = idf_Xkp1;
			int idf_Ukp1 = id0_Ukp1 + this->TrajLegs[li1].lenU;
			Eigen::VectorXd Xk = x.segment(id0_Xk, idf_Xk - id0_Xk);
			Eigen::VectorXd Uk = x.segment(id0_Uk, idf_Uk - id0_Uk);
			Eigen::VectorXd Xkp1 = x.segment(id0_Xkp1, idf_Xkp1 - id0_Xkp1);
			Eigen::VectorXd Ukp1 = x.segment(id0_Ukp1, idf_Ukp1 - id0_Ukp1);
			double T = x[this->get_dvi_T(li1)];
			Eigen::VectorXd con = LL.lfun(Xk, Uk, Xkp1, Ukp1, T, params);
			CONLK.segment(idxLK, LL.length) = con;
			idxLK += LL.length;
		}

		// Total TOF constraints.
		CONIN(idxIN) = this->minTOF - tofTOT;
		idxIN++;
		CONIN(idxIN) = tofTOT - this->maxTOF;
		idxIN++;

		Eigen::VectorXd out(1 + this->get_nec() + this->get_nic());
		out << OBJVAL, CONEQ, CONLK, CONIN;
		vector_double outRtn(out.data(), out.data() + out.size());
		return outRtn;
	}

	inline vector_double gradient(vector_double dv)
	{
		if (this->use_estimate_grad)
		{
			vector_double x = dv;
			double dx = this->est_grad_dt;
			vector_double f0 = this->fitness(x);
			vector_double grad(f0.size() * x.size(), 0.);
			vector_double x_r = x, x_l = x;
			// We change one by one each variable by dx and estimate the derivative
			for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
				double h = std::max(std::abs(x[j]), 1.0) * dx;
				x_r[j] = x[j] + h;
				x_l[j] = x[j] - h;
				vector_double f_r = this->fitness(x_r);
				vector_double f_l = this->fitness(x_l);
				if (f_r.size() != f0.size() || f_l.size() != f0.size()) {
					pagmo_throw(std::invalid_argument, "Change in the size of the returned vector detected around the "
						"reference point. Cannot compute a gradient");
				}
				for (decltype(f_r.size()) i = 0u; i < f_r.size(); ++i) {
					grad[j + i * x.size()] = (f_r[i] - f_l[i]) / 2. / h;
				}
				x_r[j] = x[j];
				x_l[j] = x[j];
			}
			return grad;
		}
		else
		{
			// Track through index.
			int idrow = 0;

			Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> x(dv.data(), dv.size());

			// Set up gradient size.
			Eigen::MatrixXd grad_clc = Eigen::MatrixXd::Zero(1 + this->get_nec() + this->get_nic(), x.size());

			// Calculate partial for cost function.
			double T0 = 0.0;
			for (int ii = 0; ii < this->idxLegObj; ii++)
			{
				T0 += x[this->get_dvi_T(ii)];
			}
			TrajLeg TLobj = this->TrajLegs[this->idxLegObj];
			TrajLeg TL = TLobj;
			int idT = this->get_dvi_T(this->idxLegObj);
			double dt = x[idT] / static_cast<double>(TL.num_nodes - 1);
			double J = this->calc_J(x);
			Eigen::VectorXd dJ = Eigen::VectorXd::Zero(x.size());
			if (TLobj.J.objType == ObjectiveFlags::LAGRANGE)
			{
				dJ(idT) = J / x[idT];
				Eigen::VectorXd Xk(TL.lenX);
				Eigen::VectorXd Uk(TL.lenU);
				std::vector<Eigen::MatrixXd> dfJ = { MatrixXd::Zero(TL.lenX, TL.lenX), MatrixXd::Zero(TL.lenX, TL.lenU) };
				vector_double params = TL.J.params;
				int id0_Xk = this->get_dvi_N(this->idxLegObj, 0);
				for (int jj = 0; jj < TLobj.num_nodes; jj++)
				{
					double Tk = T0 + dt * static_cast<double>(jj);
					int id0_Uk = id0_Xk + TL.lenX;
					Xk = x.segment(id0_Xk, TL.lenX);
					Uk = x.segment(id0_Uk, TL.lenU);
					double mlt = (jj != 0 && jj != TL.num_nodes - 1) ? 2.0 : 1.0;
					dfJ = TL.J.df(Xk, Uk, Tk, params);
					dJ.segment(id0_Xk, TL.lenX) = mlt * 0.5 * dt * dfJ[0];
					dJ.segment(id0_Uk, TL.lenU) = mlt * 0.5 * dt * dfJ[1];
					id0_Xk += TL.lenN;
				}
			}
			grad_clc.row(0) = dJ;
			idrow = 1;

			// Equality Constraints
			for (int ii = 0; ii < this->TrajLegs.size(); ii++)
			{
				TrajLeg TL = this->TrajLegs[ii];
				double T0 = 0.0;
				for (int jj = 0; jj < ii; jj++)
				{
					T0 += x[this->get_dvi_T(jj)];
				}
				int idT = this->get_dvi_T(ii);
				double dt = x[idT] / static_cast<double>(TL.num_nodes - 1);

				// Equality constraints
				for (int jj = 0; jj < TL.coneqs_f.size(); jj++)
				{
					Constraint coneq = TL.coneqs_f[jj];
					int id0 = this->get_dvi_N(ii, 0);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0;
					vector_double params = coneq.params;
					std::vector<Eigen::MatrixXd> dcon = coneq.dcon(X, U, T, params);
					grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = dcon[0];
					grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = dcon[1];
					grad_clc.block(idrow, idT, coneq.lcon, 1) = dcon[2];
					idrow += coneq.lcon;
				}
				for (int jj = 0; jj < TL.coneqs_b.size(); jj++)
				{
					Constraint coneq = TL.coneqs_b[jj];
					int id0 = this->get_dvi_N(ii, -1);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + x[this->get_dvi_T(ii)];
					vector_double params = coneq.params;
					std::vector<Eigen::MatrixXd> dcon = coneq.dcon(X, U, T, params);
					grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = dcon[0];
					grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = dcon[1];
					grad_clc.block(idrow, idT, coneq.lcon, 1) = dcon[2];
					idrow += coneq.lcon;
				}
				for (int jj = 0; jj < TL.coneqs_p.size(); jj++)
				{
					Constraint coneq = TL.coneqs_p[jj];
					vector_double params = coneq.params;
					std::vector<Eigen::MatrixXd> dcon = { MatrixXd::Zero(coneq.lcon, TL.lenX), MatrixXd::Zero(coneq.lcon, TL.lenU), MatrixXd::Zero(coneq.lcon, 1) };
					Eigen::VectorXd X(TL.lenX);
					Eigen::VectorXd U(TL.lenU);
					int id0 = this->get_dvi_N(ii, 0);
					for (int kk = 0; kk < TL.num_nodes; kk++)
					{
						X = x.segment(id0, TL.lenX);
						U = x.segment(id0 + TL.lenX, TL.lenU);
						double T = T0 + dt * static_cast<double>(kk);
						dcon = coneq.dcon(X, U, T, params);
						grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = dcon[0];
						grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = dcon[1];
						grad_clc.block(idrow, idT, coneq.lcon, 1) = dcon[2];
						idrow += coneq.lcon;
						id0 += TL.lenN;
					}
				}

				// Collocation constraints
				double nn = static_cast<double>(TL.num_nodes);
				vector_double params = TL.f.params;
				Eigen::VectorXd Xk(TL.lenX);
				Eigen::VectorXd Uk(TL.lenU);
				Eigen::VectorXd Xkp1(TL.lenX);
				Eigen::VectorXd Ukp1(TL.lenU);
				Eigen::VectorXd fk(TL.lenX);
				Eigen::VectorXd fkp1(TL.lenX);
				Eigen::VectorXd Uc(TL.lenU);
				Eigen::VectorXd Xc(TL.lenX);
				Eigen::VectorXd fc(TL.lenX);
				Eigen::MatrixXd Ak(TL.lenX, TL.lenX);
				Eigen::MatrixXd Ac(TL.lenX, TL.lenX);
				Eigen::MatrixXd Akp1(TL.lenX, TL.lenX);
				Eigen::MatrixXd Bk(TL.lenX, TL.lenX);
				Eigen::MatrixXd Bc(TL.lenX, TL.lenX);
				Eigen::MatrixXd Bkp1(TL.lenX, TL.lenX);
				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(TL.lenX, TL.lenX);
				Eigen::MatrixXd dDel_dXk(TL.lenX, TL.lenX);
				Eigen::MatrixXd dDel_dXkp1(TL.lenX, TL.lenX);
				Eigen::MatrixXd dDel_dUk(TL.lenX, TL.lenU);
				Eigen::MatrixXd dDel_dUkp1(TL.lenX, TL.lenU);
				Eigen::MatrixXd dDel_dT(TL.lenX, 1);
				std::vector<Eigen::MatrixXd> dfk = { MatrixXd::Zero(TL.lenX, TL.lenX), MatrixXd::Zero(TL.lenX, TL.lenU) };
				std::vector<Eigen::MatrixXd> dfkp1 = { MatrixXd::Zero(TL.lenX, TL.lenX), MatrixXd::Zero(TL.lenX, TL.lenU) };
				std::vector<Eigen::MatrixXd> dfc = { MatrixXd::Zero(TL.lenX, TL.lenX), MatrixXd::Zero(TL.lenX, TL.lenU) };
				int id0_Xk = this->get_dvi_N(ii, 0);
				for (int jj = 0; jj < TL.num_nodes - 1; jj++)
				{
					double Tk = T0 + dt * static_cast<double>(jj);
					double Tkp1 = T0 + dt * static_cast<double>(jj + 1);
					int id0_Uk = id0_Xk + TL.lenX;
					int id0_Xkp1 = id0_Uk + TL.lenU;
					int id0_Ukp1 = id0_Xkp1 + TL.lenX;
					Xk = x.segment(id0_Xk, TL.lenX);
					Uk = x.segment(id0_Uk, TL.lenU);
					Xkp1 = x.segment(id0_Xkp1, TL.lenX);
					Ukp1 = x.segment(id0_Ukp1, TL.lenU);
					fk = TL.f.f(Xk, Uk, Tk, params);
					fkp1 = TL.f.f(Xkp1, Ukp1, Tkp1, params);
					Uc = 0.5 * (Uk + Ukp1);
					Xc = 0.5 * (Xk + Xkp1) + dt / 8 * (fk - fkp1);
					double Tc = 0.5 * (Tk + Tkp1);
					fc = TL.f.f(Xc, Uc, Tc, params);
					dfk = TL.f.df(Xk, Uk, Tk, params);
					dfkp1 = TL.f.df(Xkp1, Ukp1, Tkp1, params);
					dfc = TL.f.df(Xc, Uc, Tc, params);
					Ak = dfk[0];
					Akp1 = dfkp1[0];
					Ac = dfc[0];
					Bk = dfk[1];
					Bkp1 = dfkp1[1];
					Bc = dfc[1];
					dDel_dXk = I + dt / 6.0 * (Ak + 4.0 * Ac * (0.5 * I + dt / 8.0 * Ak));
					dDel_dXkp1 = -I + dt / 6.0 * (Akp1 + 4.0 * Ac * (0.5 * I - dt / 8.0 * Akp1));
					dDel_dUk = dt / 6.0 * (Bk + 4.0 * (Ac * (dt / 8.0 * Bk) + 1.0 / 2.0 * Bc));
					dDel_dUkp1 = dt / 6.0 * (Bkp1 + 4.0 * (Ac * (-dt / 8.0 * Bkp1) + 1.0 / 2.0 * Bc));
					dDel_dT = 1.0 / (6.0 * (nn - 1.0)) * (fk + 4.0 * fc + fkp1) + dt / 6.0 * (Ac * (1.0 / (2.0 * (nn - 1.0))) * (fk - fkp1));
					grad_clc.block(idrow, id0_Xk, TL.lenX, TL.lenX) = dDel_dXk;
					grad_clc.block(idrow, id0_Uk, TL.lenX, TL.lenU) = dDel_dUk;
					grad_clc.block(idrow, id0_Xkp1, TL.lenX, TL.lenX) = dDel_dXkp1;
					grad_clc.block(idrow, id0_Ukp1, TL.lenX, TL.lenU) = dDel_dUkp1;
					grad_clc.block(idrow, idT, TL.lenX, 1) = dDel_dT;
					idrow += TL.lenX;
					id0_Xk += TL.lenN;
				}
			}

			// Linking constraints
			for (int ii = 0; ii < this->LegLinks.size(); ii++)
			{
				LegLink LL = this->LegLinks[ii];
				int li0 = LL.l1;
				int li1 = LL.l2;
				vector_double params = LL.params;
				int id0_Xk = this->get_dvi_N(li0, -1);
				int idf_Xk = id0_Xk + this->TrajLegs[li0].lenX;
				int id0_Uk = idf_Xk;
				int idf_Uk = id0_Uk + this->TrajLegs[li0].lenU;
				int id0_Xkp1 = this->get_dvi_N(li1, 0);
				int idf_Xkp1 = id0_Xkp1 + this->TrajLegs[li1].lenX;
				int id0_Ukp1 = idf_Xkp1;
				int idf_Ukp1 = id0_Ukp1 + this->TrajLegs[li1].lenU;
				Eigen::VectorXd Xk = x.segment(id0_Xk, idf_Xk - id0_Xk);
				Eigen::VectorXd Uk = x.segment(id0_Uk, idf_Uk - id0_Uk);
				Eigen::VectorXd Xkp1 = x.segment(id0_Xkp1, idf_Xkp1 - id0_Xkp1);
				Eigen::VectorXd Ukp1 = x.segment(id0_Ukp1, idf_Ukp1 - id0_Ukp1);
				double idT0 = this->get_dvi_T(li0);
				double idT1 = this->get_dvi_T(li1);
				std::vector<Eigen::MatrixXd> dls = LL.dlfun(Xk, Uk, Xkp1, Ukp1, x[idT1], params);
				Eigen::MatrixXd dl1dX = dls[0];
				Eigen::MatrixXd dl1dU = dls[1];
				Eigen::MatrixXd dl1dT = dls[2];
				Eigen::MatrixXd dl2dX = dls[3];
				Eigen::MatrixXd dl2dU = dls[4];
				Eigen::MatrixXd dl2dT = dls[5];
				grad_clc.block(idrow, id0_Xk, LL.length, this->TrajLegs[li0].lenX) = dl1dX;
				grad_clc.block(idrow, id0_Uk, LL.length, this->TrajLegs[li0].lenU) = dl1dU;
				grad_clc.block(idrow, idT0, LL.length, 1) = dl1dT;
				grad_clc.block(idrow, id0_Xkp1, LL.length, this->TrajLegs[li1].lenX) = dl2dX;
				grad_clc.block(idrow, id0_Ukp1, LL.length, this->TrajLegs[li1].lenU) = dl2dU;
				grad_clc.block(idrow, idT1, LL.length, 1) = dl2dT;
				idrow += LL.length;
			}

			// Inequality Constraints
			for (int ii = 0; ii < this->TrajLegs.size(); ii++)
			{
				TrajLeg TL = this->TrajLegs[ii];
				double T0 = 0.0;
				for (int jj = 0; jj < ii; jj++)
				{
					T0 += x[this->get_dvi_T(jj)];
				}
				int idT = this->get_dvi_T(ii);
				double dt = x[idT] / static_cast<double>(TL.num_nodes - 1);

				// Inequality constraints
				for (int jj = 0; jj < TL.conins_f.size(); jj++)
				{
					Constraint conin = TL.conins_f[jj];
					int id0 = this->get_dvi_N(ii, 0);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0;
					vector_double params = conin.params;
					std::vector<Eigen::MatrixXd> dcon = conin.dcon(X, U, T, params);
					grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = dcon[0];
					grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = dcon[1];
					grad_clc.block(idrow, idT, conin.lcon, 1) = dcon[2];
					idrow += conin.lcon;
				}
				for (int jj = 0; jj < TL.conins_b.size(); jj++)
				{
					Constraint conin = TL.conins_b[jj];
					int id0 = this->get_dvi_N(ii, -1);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + x[this->get_dvi_T(ii)];
					vector_double params = conin.params;
					std::vector<Eigen::MatrixXd> dcon = conin.dcon(X, U, T, params);
					grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = dcon[0];
					grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = dcon[1];
					grad_clc.block(idrow, idT, conin.lcon, 1) = dcon[2];
					idrow += conin.lcon;
				}
				for (int jj = 0; jj < TL.conins_p.size(); jj++)
				{
					Constraint conin = TL.conins_p[jj];
					vector_double params = conin.params;
					int id0 = this->get_dvi_N(ii, 0);
					for (int kk = 0; kk < TL.num_nodes; kk++)
					{
						Eigen::VectorXd X = x.segment(id0, TL.lenX);
						Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
						double T = T0 + dt * static_cast<double>(kk);
						std::vector<Eigen::MatrixXd> dcon = conin.dcon(X, U, T, params);
						grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = dcon[0];
						grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = dcon[1];
						grad_clc.block(idrow, idT, conin.lcon, 1) = dcon[2];
						idrow += conin.lcon;
						id0 += TL.lenN;
					}
				}

				// Leg TOF constraints
				grad_clc(idrow, idT) = -1.0;
				idrow++;
				grad_clc(idrow, idT) = 1.0;
				idrow++;
			}

			// Total TOF constraints.
			for (int ii = 0; ii < this->TrajLegs.size(); ii++)
			{
				TrajLeg TL = this->TrajLegs[ii];
				int idT = this->get_dvi_T(ii);
				grad_clc(idrow, idT) = -1.0;
				grad_clc(idrow + 1, idT) = 1.0;
			}
			idrow += 2;

			vector_double sparseOut(this->sparse_pattern.size());
			for (int ii = 0; ii < this->sparse_pattern.size(); ii++)
			{
				int id0 = this->sparse_pattern[ii].first;
				int id1 = this->sparse_pattern[ii].second;
				sparseOut[ii] = (grad_clc(id0, id1));
			}

			return sparseOut;
		}
	}




	void calc_gradient_sparsity(vector_double dv)
	{
		using Vec = Eigen::VectorXd;
		using Mat = Eigen::MatrixXd;

		// Track through index.
		int idrow = 0;

		Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> x(dv.data(), dv.size());

		// Set up gradient size.
		Eigen::MatrixXd grad_clc = Eigen::MatrixXd::Zero(1 + this->get_nec() + this->get_nic(), x.size());

		// Calculate partial for cost function.
		TrajLeg TLobj = this->TrajLegs[this->idxLegObj];
		TrajLeg TL = TLobj;
		int idT = this->get_dvi_T(this->idxLegObj);
		Eigen::VectorXd dJ = Eigen::VectorXd::Zero(x.size());
		if (TLobj.J.objType == ObjectiveFlags::LAGRANGE)
		{
			dJ(idT) = 1;
			for (int jj = 0; jj < TLobj.num_nodes; jj++)
			{
				int id0_Xk = this->get_dvi_N(this->idxLegObj, jj);
				int id0_Uk = id0_Xk + TL.lenX;
				dJ.segment(id0_Xk, TL.lenX) = Vec::Ones(TL.lenX);
				dJ.segment(id0_Uk, TL.lenU) = Vec::Ones(TL.lenU);
			}
		}
		grad_clc.row(0) = dJ;
		idrow = 1;

		// Equality Constraints
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			int idT = this->get_dvi_T(ii);

			// Equality constraints
			for (int jj = 0; jj < TL.coneqs_f.size(); jj++)
			{
				Constraint coneq = TL.coneqs_f[jj];
				int id0 = this->get_dvi_N(ii, 0);
				grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = Mat::Ones(coneq.lcon, TL.lenX);
				grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = Mat::Ones(coneq.lcon, TL.lenU);
				grad_clc.block(idrow, idT, coneq.lcon, 1) = Mat::Ones(coneq.lcon, 1);
				idrow += coneq.lcon;
			}
			for (int jj = 0; jj < TL.coneqs_b.size(); jj++)
			{
				Constraint coneq = TL.coneqs_b[jj];
				int id0 = this->get_dvi_N(ii, -1);
				grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = Mat::Ones(coneq.lcon, TL.lenX);
				grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = Mat::Ones(coneq.lcon, TL.lenU);
				grad_clc.block(idrow, idT, coneq.lcon, 1) = Mat::Ones(coneq.lcon, 1);
				idrow += coneq.lcon;
			}
			for (int jj = 0; jj < TL.coneqs_p.size(); jj++)
			{
				Constraint coneq = TL.coneqs_p[jj];
				vector_double params = coneq.params;
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					int id0 = this->get_dvi_N(ii, kk);
					grad_clc.block(idrow, id0, coneq.lcon, TL.lenX) = Mat::Ones(coneq.lcon, TL.lenX);
					grad_clc.block(idrow, id0 + TL.lenX, coneq.lcon, TL.lenU) = Mat::Ones(coneq.lcon, TL.lenU);
					grad_clc.block(idrow, idT, coneq.lcon, 1) = Mat::Ones(coneq.lcon, 1);
					idrow += coneq.lcon;
				}
			}

			// Collocation constraints
			for (int jj = 0; jj < TL.num_nodes - 1; jj++)
			{
				int id0_Xk = this->get_dvi_N(ii, jj);
				int id0_Uk = id0_Xk + TL.lenX;
				int id0_Xkp1 = id0_Uk + TL.lenU;
				int id0_Ukp1 = id0_Xkp1 + TL.lenX;
				grad_clc.block(idrow, id0_Xk, TL.lenX, TL.lenX) = Mat::Ones(TL.lenX, TL.lenX);
				grad_clc.block(idrow, id0_Uk, TL.lenX, TL.lenU) = Mat::Ones(TL.lenX, TL.lenU);
				grad_clc.block(idrow, id0_Xkp1, TL.lenX, TL.lenX) = Mat::Ones(TL.lenX, TL.lenX);
				grad_clc.block(idrow, id0_Ukp1, TL.lenX, TL.lenU) = Mat::Ones(TL.lenX, TL.lenU);
				grad_clc.block(idrow, idT, TL.lenX, 1) = Mat::Ones(TL.lenX, 1);
				idrow += TL.lenX;
			}
		}

		// Linking constraints
		for (int ii = 0; ii < this->LegLinks.size(); ii++)
		{
			LegLink LL = this->LegLinks[ii];
			int li0 = LL.l1;
			int li1 = LL.l2;
			int id0_Xk = this->get_dvi_N(li0, -1);
			int idf_Xk = id0_Xk + this->TrajLegs[li0].lenX;
			int id0_Uk = idf_Xk;
			int idf_Uk = id0_Uk + this->TrajLegs[li0].lenU;
			int id0_Xkp1 = this->get_dvi_N(li1, 0);
			int idf_Xkp1 = id0_Xkp1 + this->TrajLegs[li1].lenX;
			int id0_Ukp1 = idf_Xkp1;
			int idf_Ukp1 = id0_Ukp1 + this->TrajLegs[li1].lenU;
			double idT0 = this->get_dvi_T(li0);
			double idT1 = this->get_dvi_T(li1);
			grad_clc.block(idrow, id0_Xk, LL.length, this->TrajLegs[li0].lenX) = Mat::Ones(LL.length, this->TrajLegs[li0].lenX);
			grad_clc.block(idrow, id0_Uk, LL.length, this->TrajLegs[li0].lenU) = Mat::Ones(LL.length, this->TrajLegs[li0].lenU);
			grad_clc.block(idrow, idT0, LL.length, 1) = Mat::Ones(LL.length, 1);
			grad_clc.block(idrow, id0_Xkp1, LL.length, this->TrajLegs[li1].lenX) = Mat::Ones(LL.length, this->TrajLegs[li0].lenX);
			grad_clc.block(idrow, id0_Ukp1, LL.length, this->TrajLegs[li1].lenU) = Mat::Ones(LL.length, this->TrajLegs[li0].lenU);
			grad_clc.block(idrow, idT1, LL.length, 1) = Mat::Ones(LL.length, 1);
			idrow += LL.length;
		}

		// Inequality Constraints
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			int idT = this->get_dvi_T(ii);

			// Inequality constraints
			for (int jj = 0; jj < TL.conins_f.size(); jj++)
			{
				Constraint conin = TL.conins_f[jj];
				int id0 = this->get_dvi_N(ii, 0);
				grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = Mat::Ones(conin.lcon, TL.lenX);
				grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = Mat::Ones(conin.lcon, TL.lenU);
				grad_clc.block(idrow, idT, conin.lcon, 1) = Mat::Ones(conin.lcon, 1);
				idrow += conin.lcon;
			}
			for (int jj = 0; jj < TL.conins_b.size(); jj++)
			{
				Constraint conin = TL.conins_b[jj];
				int id0 = this->get_dvi_N(ii, -1);
				grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = Mat::Ones(conin.lcon, TL.lenX);
				grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = Mat::Ones(conin.lcon, TL.lenU);
				grad_clc.block(idrow, idT, conin.lcon, 1) = Mat::Ones(conin.lcon, 1);
				idrow += conin.lcon;
			}
			for (int jj = 0; jj < TL.conins_p.size(); jj++)
			{
				Constraint conin = TL.conins_p[jj];
				vector_double params = conin.params;
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					int id0 = this->get_dvi_N(ii, kk);
					grad_clc.block(idrow, id0, conin.lcon, TL.lenX) = Mat::Ones(conin.lcon, TL.lenX);
					grad_clc.block(idrow, id0 + TL.lenX, conin.lcon, TL.lenU) = Mat::Ones(conin.lcon, TL.lenU);
					grad_clc.block(idrow, idT, conin.lcon, 1) = Mat::Ones(conin.lcon, 1);
					idrow += conin.lcon;
				}
			}

			// Leg TOF constraints
			grad_clc(idrow, idT) = 1;
			idrow++;
			grad_clc(idrow, idT) = 1;
			idrow++;
		}

		// Total TOF constraints.
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			TrajLeg TL = this->TrajLegs[ii];
			int idT = this->get_dvi_T(ii);
			grad_clc(idrow, idT) = 1;
			grad_clc(idrow + 1, idT) = 1;
		}

		std::vector<std::pair<vector_double::size_type, vector_double::size_type>> sparsityPattern;
		for (int ii = 0; ii < grad_clc.rows(); ii++)
		{
			for (int jj = 0; jj < grad_clc.cols(); jj++)
			{
				if (fabs(grad_clc(ii, jj)) > 0.0)
				{
					sparsityPattern.push_back(std::make_pair(ii, jj));
				}
			}
		}
		this->sparse_pattern = sparsityPattern;
		
		return;
	}

	sparsity_pattern gradient_sparsity()
	{
		return this->sparse_pattern;
	}

	void add_leg(TrajLeg tl) 
	{
		TrajLegs.push_back(tl);
		if (tl.objSet)
		{
			this->idxLegObj = this->TrajLegs.size() - 1;
		}
	}

	inline int getTotLength()
	{
		int len = 0;
		for (int ii = 0; ii < this->TrajLegs.size(); ii++)
		{
			len += this->TrajLegs[ii].getTotLength();
		}
		return len;
	}

	inline int get_dvi_L(int idx_leg)
	{
		int i0 = 0;
		if (idx_leg == -1)
		{
			idx_leg = this->TrajLegs.size() - 1;
		}
		for (int ii = 0; ii < idx_leg; ii++)
		{
			i0 = i0 + this->TrajLegs[ii].getTotLength();
		}
		return i0;
	}

	inline int get_dvi_N(int idx_leg, int idx_node)
	{
		int i0 = this->get_dvi_L(idx_leg);
		i0 += 1; // Compensate for time
		if (idx_node == -1)
		{
			idx_node = this->TrajLegs[idx_leg].num_nodes - 1;
		}
		for (int ii = 0; ii < idx_node; ii++)
		{
			i0 = i0 + this->TrajLegs[idx_leg].lenN;
		}
		return i0;
	}

	inline int get_dvi_U(int idx_leg, int idx_node)
	{
		int i0 = this->get_dvi_N(idx_leg, idx_node);
		i0 = i0 + this->TrajLegs[idx_leg].lenX;
		return i0;
	}

	inline int get_dvi_X(int idx_leg, int idx_node) 
	{
		int i0 = this->get_dvi_N(idx_leg, idx_node);
		return i0;
	}

	inline int get_dvi_T(int idx_leg)
	{
		int i0 = this->get_dvi_L(idx_leg);
		return i0;
	}

	static void Build(py::module& m)
	{
		auto obj = py::class_<Ananke_Config>(m, "Ananke_Config");
		obj.def(py::init<>());
		obj.def_readwrite("TrajLegs", &Ananke_Config::TrajLegs);
		obj.def_readwrite("LegLinks", &Ananke_Config::LegLinks);
		obj.def_readwrite("maxTOF", &Ananke_Config::maxTOF);
		obj.def_readwrite("minTOF", &Ananke_Config::minTOF);
		obj.def_readwrite("idxLegObj", &Ananke_Config::idxLegObj);
		obj.def_readwrite("use_estimate_grad", &Ananke_Config::use_estimate_grad);
		obj.def_readwrite("est_grad_dt", &Ananke_Config::est_grad_dt);

		obj.def("add_leg_link", &Ananke_Config::add_leg_link);
		obj.def("set_TOF", &Ananke_Config::set_TOF);
		obj.def("get_array_data", &Ananke_Config::get_array_data);
		obj.def("add_leg", &Ananke_Config::add_leg);
		obj.def("getTotLength", &Ananke_Config::getTotLength);
		obj.def("get_dvi_L", &Ananke_Config::get_dvi_L);
		obj.def("get_dvi_N", &Ananke_Config::get_dvi_N);
		obj.def("get_dvi_U", &Ananke_Config::get_dvi_U);
		obj.def("get_dvi_X", &Ananke_Config::get_dvi_X);
		obj.def("get_dvi_T", &Ananke_Config::get_dvi_T);

	}
};