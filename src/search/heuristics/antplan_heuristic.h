#ifndef HEURISTICS_ANTPLAN_HEURISTIC_H
#define HEURISTICS_ANTPLAN_HEURISTIC_H

#include "additive_heuristic.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace antplan_heuristic {

// Pull the relaxation_heuristic types into this namespace so the .cc
// can use them unqualified (fixes "PropID has not been declared", etc.).
using relaxation_heuristic::PropID;
using relaxation_heuristic::OpID;
using relaxation_heuristic::NO_OP;
using relaxation_heuristic::Proposition;
using relaxation_heuristic::UnaryOperator;

/*
 * AntPlanHeuristic
 *
 * Calls a Python cost function on each state and converts the returned
 * float into a non-negative integer suitable for Fast Downward's search.
 *
 * Conversion formula:
 *     h = clamp( round(raw_cost * scale_factor) + offset, 0, MAX_H )
 *
 * This preserves ordering of negative floats.  For example, with
 * scale_factor=10000 and offset=100000:
 *     -0.4130  ->  round(-4130) + 100000 = 95870
 *     -0.0884  ->  round(-884)  + 100000 = 99116
 * So -0.4130 correctly maps to a *lower* heuristic value (= preferred).
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
class AntPlanHeuristic
    : public additive_heuristic::AdditiveHeuristic {
public:
    explicit AntPlanHeuristic(const options::Options &opts);
    ~AntPlanHeuristic() override;

    /// (Re-)bind to a different Python function at runtime.
    static void initialize_python_function(const std::string &func_name);

protected:
    int compute_heuristic(const State &ancestor_state) override;

private:
    // ---- Python binding (static, shared across instances) ----
    static py::object  py_cost_fn_;
    static bool        py_ready_;
    static std::string py_func_name_;
    static std::string py_module_name_;

    void ensure_python_ready();

    // =====================================================================
    // IMPORTANT: members are declared in EXACTLY the order the constructor
    // initialiser-list uses them.  The compiler initialises in declaration
    // order, not init-list order, and -Wreorder will fire otherwise.
    // =====================================================================

    // ---- Relaxed-plan / preferred operators ----
    std::vector<bool> relaxed_plan_;

    // ---- Float -> int conversion ----
    double scale_factor_;
    int    offset_;
    static constexpr int MAX_H = 100000000;

    int float_to_heuristic(double raw) const;

    // ---- Optional memoisation cache ----
    bool   use_cache_;
    size_t cache_max_entries_;
    std::unordered_map<uint64_t, int> cache_;

    uint64_t hash_state(const State &state) const;
    void     maybe_evict_cache();

    // ---- Debug / stats ----
    bool     debug_;
    bool     log_states_;
    uint64_t stat_calls_      = 0;
    uint64_t stat_cache_hits_ = 0;
    uint64_t stat_cache_miss_ = 0;

    // ---- Pre-built Python string tables ----
    bool                              tables_ready_ = false;
    std::vector<py::str>              var_names_;
    std::vector<std::vector<py::str>> fact_names_;
    void ensure_tables(TaskProxy &tp);

    void mark_preferred_operators_and_relaxed_plan(
        const State &state, PropID goal_id);
};
#pragma GCC diagnostic pop

}  // namespace antplan_heuristic

#endif  // HEURISTICS_ANTPLAN_HEURISTIC_H