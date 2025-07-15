#ifndef SEARCH_HEURISTICS_ANTPLAN_HEURISTIC_H
#define SEARCH_HEURISTICS_ANTPLAN_HEURISTIC_H

#include "ff_heuristic.h"
#include <pybind11/embed.h>
#include <string>

namespace py = pybind11;

namespace antplan_heuristic {

class AntPlanHeuristic : public ff_heuristic::FFHeuristic {
private:
    static py::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;

    static void ensure_python_ready();
    int compute_antplan_cost(const State &state);

public:
    explicit AntPlanHeuristic(const options::Options &opts);
    virtual ~AntPlanHeuristic();

    virtual int compute_heuristic(const State &ancestor_state) override;

    static void initialize_python_function(const std::string &func_name);
};

}

#endif
