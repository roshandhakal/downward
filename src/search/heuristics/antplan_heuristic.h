#ifndef ANTPLAN_HEURISTIC_H
#define ANTPLAN_HEURISTIC_H

#include "../heuristics/additive_heuristic.h"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string>
#include <unordered_set>
#include <cstdint>

namespace py = pybind11;

namespace antplan_heuristic {

class AntPlanHeuristic : public additive_heuristic::AdditiveHeuristic {
private:
    // Python function info
    static py::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;
    static std::string py_module_name;

    // Exploration parameters
    int exploration_frequency;
    int exploration_depth;
    double improvement_threshold;
    int exploration_budget;

    // Tracking
    static int evaluation_count;
    static int exploration_count;
    static std::unordered_set<uint64_t> explored_states;

    // Relaxed plan tracking (from base class usage)
    std::vector<bool> relaxed_plan;

    // Helper methods
    void ensure_python_ready();
    bool should_explore_now();
    void explore_from_state(const State &state, int current_cost);
    void probe_successors(const State &state, int current_cost, int depth, int &budget);
    double evaluate_state_with_nn(const State &state);

protected:
    virtual int compute_heuristic(const State &ancestor_state) override;

    virtual void mark_preferred_operators_and_relaxed_plan(
        const State &state, relaxation_heuristic::PropID goal_id);

public:
    explicit AntPlanHeuristic(const options::Options &opts);
    virtual ~AntPlanHeuristic() override;
};

}

#endif