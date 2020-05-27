#include <pybind11/pybind11.h>
#include "ex1_setup.h"
#include "TrajLeg.h"
#include "Ananke_Config.h"


PYBIND11_MODULE(AnankeC, m) {
    m.doc() = "AnankeC plugin"; // optional module docstring
    
    TrajLeg::Build(m);

    Ananke_Config::Build(m);

    Ex1::Build(m);

    m.def("optimize", optimize);
    m.def("set_ac", set_ac);
    m.def("set_dv", set_dv);
}