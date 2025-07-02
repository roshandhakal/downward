/************************  antplan_heuristic.h  ************************
 * A Fast Downward heuristic that forwards each state to a Python
 * function `anticipatory_cost_fn(state_vals:list[int]) -> int`.
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

protected:
    /** compute_heuristic is called by the search engine for each state. */
    int compute_heuristic(const State &ancestor_state) override;

private:
    /** One-time Python interpreter (pybind11) bootstrap. */
    static void ensure_python_ready();

    /** Cached handle to the Python callback. */
    static inline bool              py_ready  = false;
    static inline pybind11::object  py_cost_fn;
};

}  // namespace antplan_heuristic
