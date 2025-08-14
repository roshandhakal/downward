#ifndef HEURISTICS_ANTPLAN_HEURISTIC_H
#define HEURISTICS_ANTPLAN_HEURISTIC_H

#include "additive_heuristic.h"
#include "../task_proxy.h"

#include <map>
#include <string>
#include <vector>

#include <pybind11/embed.h>

namespace py = pybind11;

namespace antplan_heuristic {
using relaxation_heuristic::NO_OP;
using relaxation_heuristic::OpID;
using relaxation_heuristic::PropID;
using relaxation_heuristic::Proposition;
using relaxation_heuristic::UnaryOperator;

class AntPlanHeuristic : public additive_heuristic::AdditiveHeuristic {
    using RelaxedPlan = std::vector<bool>;
    RelaxedPlan relaxed_plan;

    // Python integration
    static py::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;
    static std::string py_module_name; // module to import if filepath is empty
    static std::string py_file_path;   // .py file path; takes precedence if set
    static std::string py_sys_path;    // optional extra sys.path[0] entry

    void initialize_python_function(const std::string &func_name);
    void ensure_python_ready();

    static void add_to_sys_path_front(const std::string &path);

    std::map<std::string, std::string> convert_state_to_map(const State &state);
    void mark_preferred_operators_and_relaxed_plan(const State &state, PropID goal_id);

protected:
    int compute_heuristic(const State &ancestor_state) override;

public:
    explicit AntPlanHeuristic(const options::Options &opts);
    ~AntPlanHeuristic() override;
};

} // namespace antplan_heuristic

#endif
