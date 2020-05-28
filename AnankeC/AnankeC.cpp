#include <pybind11/pybind11.h>
#include "ex1_setup.h"
#include "TrajLeg.h"
#include "Ananke_Config.h"

py::tuple optimize(int max_eval, int verb)
{
	// Construct a pagmo::problem from our example problem.
	problem p { Ananke_Problem{} };
	algorithm algo{ nlopt("slsqp") };
	algo.set_verbosity(verb);
	p.set_c_tol(1e-4);
	algo.extract<nlopt>()->set_maxeval(max_eval);
	algo.extract<nlopt>()->set_ftol_rel(0.0);
	algo.extract<nlopt>()->set_xtol_rel(0.0);
	population pop{ p };
	pop.push_back(X0);

	pop = algo.evolve(pop);

	return py::make_tuple(pop.champion_x(), pop.champion_f());
}


PYBIND11_MODULE(AnankeC, m) {
    m.doc() = "AnankeC plugin"; // optional module docstring
           
    Ex1::Build(m);
    TrajLeg::Build(m);

    Ananke_Config::Build(m);
    m.def("optimize", optimize);
    m.def("set_ac", set_ac);
    m.def("set_dv", set_dv);
}