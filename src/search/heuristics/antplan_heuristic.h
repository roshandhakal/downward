#ifndef SEARCH_HEURISTICS_ANTPLAN_HEURISTIC_H
#define SEARCH_HEURISTICS_ANTPLAN_HEURISTIC_H

#include "relaxation_heuristic.h"
#include "../option_parser.h"
#include "../plugin.h"

#include <pybind11/pybind11.h>
#include <queue>
#include <vector>
#include <map>
#include <string>

namespace antplan_heuristic {

using relaxation_heuristic::PropID;

struct QueueEntry {
    int combined_cost;
    PropID prop_id;

    bool operator>(const QueueEntry &other) const {
        return combined_cost > other.combined_cost;
    }
};

class AntPlanHeuristic : public relaxation_heuristic::RelaxationHeuristic {
    static pybind11::object py_cost_fn;
    static bool py_ready;
    static std::string py_func_name;

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;

    void setup_exploration_queue();
    void setup_exploration_queue_state(const State &state, std::vector<int> *facts);
    void enqueue_if_necessary(PropID prop_id, int distance, const std::vector<int> &facts);

    void initialize_python_function(const std::string &func_name);
    void ensure_python_ready();

public:
    explicit AntPlanHeuristic(const options::Options &opts);
    virtual int compute_heuristic(const State &ancestor_state) override;
};

} // namespace antplan_heuristic

#endif
