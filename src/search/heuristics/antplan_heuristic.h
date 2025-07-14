#ifndef HEURISTICS_ANTPLAN_HEURISTIC_H
#define HEURISTICS_ANTPLAN_HEURISTIC_H

#include "../heuristic.h"
#include <memory>
#include <vector>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace antplan_heuristic {

class AntPlanHeuristic : public Heuristic {
public:
    explicit AntPlanHeuristic(const options::Options &opts);
    ~AntPlanHeuristic() override;

protected:
    int compute_heuristic(const State &ancestor_state) override;

private:
    void ensure_python_ready();
    void initialize_python_function(const std::string &func_name);

    static bool py_ready;
    static py::object py_cost_fn;
    static std::string py_func_name;
};

}

#endif
