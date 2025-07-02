/************************  antplan_heuristic.h  ************************
 * A Fast Downward heuristic that forwards each state to a Python
 * function `function(state_vals: list[int]) -> int`.
 *********************************************************************/
#pragma once

#include "../heuristic.h"          // Fast Downward base class
#include "../plugins/plugin.h"     // Option-parser helpers
#include <memory>
#include <vector>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace antplan_heuristic {

class AntPlanHeuristic : public Heuristic {
public:
    AntPlanHeuristic(const std::shared_ptr<AbstractTask> &task,
                     bool cache_estimates,
                     const std::string &description,
                     utils::Verbosity verbosity);

    // Called by plugin to set custom Python function name
    void initialize_python_function(const std::string &func_name);

protected:
    int compute_heuristic(const State &ancestor_state) override;

private:
    static void ensure_python_ready();

    static bool py_ready;
    static py::object py_cost_fn;
    static std::string py_func_name;
};

}  // namespace antplan_heuristic
