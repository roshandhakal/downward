#ifndef ANTPLAN_HEURISTIC_H
#define ANTPLAN_HEURISTIC_H

#include "relaxation_heuristic.h"
#include "../algorithms/priority_queues.h"
#include <pybind11/embed.h>
#include <vector>
#include <string>
#include <map>

namespace antplan_heuristic {

using relaxation_heuristic::PropID;
using relaxation_heuristic::UnaryOperator;
using relaxation_heuristic::Proposition;

class AntPlanHeuristic : public relaxation_heuristic::RelaxationHeuristic {
    struct RelaxedNode {
        int cost;                  // Accumulated symbolic cost
        PropID prop_id;            // Proposition ID
        std::vector<int> facts;    // Snapshot of relaxed state

        bool operator>(const RelaxedNode &other) const {
            return cost > other.cost;
        }
    };

    // Priority queue for relaxed nodes
    std::priority_queue<RelaxedNode, std::vector<RelaxedNode>, std::greater<RelaxedNode>> queue;

    static pybind11::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;

    void setup_exploration_queue();
    void setup_exploration_queue_state(const State &state);
    void enqueue_if_necessary(PropID prop_id, int cost, const std::vector<int> &facts);

    void initialize_python_function(const std::string &func_name);
    void ensure_python_ready();

public:
    explicit AntPlanHeuristic(const options::Options &opts);
    virtual int compute_heuristic(const State &ancestor_state) override;
};

} // namespace antplan_heuristic

#endif
