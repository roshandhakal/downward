#ifndef HEURISTICS_ANTPLAN_HEURISTIC_H
#define HEURISTICS_ANTPLAN_HEURISTIC_H

#include "relaxation_heuristic.h"
#include "../algorithms/priority_queues.h"

#include <pybind11/pybind11.h>
#include <string>
#include <map>

namespace antplan_heuristic {

using relaxation_heuristic::PropID;
using relaxation_heuristic::OpID;
using relaxation_heuristic::Proposition;
using relaxation_heuristic::UnaryOperator;

class AntPlanHeuristic : public relaxation_heuristic::RelaxationHeuristic {
    priority_queues::AdaptiveQueue<PropID> queue;

    void setup_exploration_queue();
    void setup_exploration_queue_state(const State &state);
    void enqueue_if_necessary(PropID prop_id, int cost);
    void initialize_python_function(const std::string &func_name);
    void ensure_python_ready();

    static pybind11::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;

protected:
    virtual int compute_heuristic(const State &ancestor_state) override;

public:
    explicit AntPlanHeuristic(const options::Options &opts);
};

} // namespace antplan_heuristic

#endif
