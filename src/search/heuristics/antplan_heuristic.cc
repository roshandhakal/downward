#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace antplan_heuristic {

// ===== Static members =====
py::object  AntPlanHeuristic::py_cost_fn_;
bool        AntPlanHeuristic::py_ready_       = false;
string      AntPlanHeuristic::py_func_name_   = "anticipatory_cost_fn";
string      AntPlanHeuristic::py_module_name_ = "antplan.scripts.eval_antplan_gripper";


// ===================================================================
//  Helpers
// ===================================================================

/// FNV-1a 64-bit update step.
static inline uint64_t fnv1a_update(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
    return h;
}

/// Extract a double from a Python object that may be a torch/numpy scalar.
static inline double py_to_double(py::object obj) {
    if (py::hasattr(obj, "item"))
        obj = obj.attr("item")();
    return obj.cast<double>();
}


// ===================================================================
//  Construction / destruction
// ===================================================================

AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      // ---- init order MUST match declaration order in .h ----
      relaxed_plan_(task_proxy.get_operators().size(), false),
      scale_factor_(opts.get<int>("scale")),
      offset_(opts.get<int>("offset")),
      use_cache_(opts.get<bool>("cache")),
      cache_max_entries_(
          static_cast<size_t>(max(0, opts.get<int>("cache_max_entries")))),
      debug_(opts.get<bool>("debug")),
      log_states_(opts.get<bool>("log_states"))
{
    py_func_name_   = opts.get<string>("function");
    py_module_name_ = opts.get<string>("module");
    py_ready_       = false;

    utils::g_log << "[AntPlan] function=" << py_func_name_
                 << "  module="
                 << (py_module_name_.empty() ? "<none>" : py_module_name_)
                 << "  scale=" << scale_factor_
                 << "  offset=" << offset_
                 << endl;

    ensure_python_ready();
    ensure_tables(task_proxy);
}

AntPlanHeuristic::~AntPlanHeuristic() {
    if (debug_) {
        utils::g_log << "[AntPlan] stats: calls=" << stat_calls_
                     << "  cache_hits=" << stat_cache_hits_
                     << "  cache_misses=" << stat_cache_miss_ << endl;
    }
}


// ===================================================================
//  Python initialisation
// ===================================================================

void AntPlanHeuristic::initialize_python_function(const string &func_name) {
    py_func_name_ = func_name;
    py_ready_     = false;
    // Caller is expected to invoke ensure_python_ready() afterwards.
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready_)
        return;

    // If the interpreter isn't running yet, start it via the C API.
    // (py::initialize_interpreter() is only available with pybind11's
    //  scoped_interpreter; the FD build embeds Python differently.)
    if (!Py_IsInitialized())
        Py_Initialize();

    py::gil_scoped_acquire gil;

    if (py_module_name_.empty())
        throw runtime_error("[AntPlan] No Python module provided.");

    // Make sure "." is on sys.path exactly once.
    py::module sys  = py::module::import("sys");
    py::list    path = sys.attr("path");
    bool has_dot = false;
    for (auto item : path)
        if (py::cast<string>(item) == ".") { has_dot = true; break; }
    if (!has_dot)
        path.attr("insert")(0, ".");

    py::object mod = py::module::import(py_module_name_.c_str());
    if (!py::hasattr(mod, py_func_name_.c_str()))
        throw runtime_error(
            "[AntPlan] '" + py_func_name_ +
            "' not found in module '" + py_module_name_ + "'");

    py_cost_fn_ = mod.attr(py_func_name_.c_str());
    py_ready_   = true;

    if (debug_)
        utils::g_log << "[AntPlan] Python ready." << endl;
}


// ===================================================================
//  Pre-built Python string tables
// ===================================================================

void AntPlanHeuristic::ensure_tables(TaskProxy &tp) {
    if (tables_ready_)
        return;

    py::gil_scoped_acquire gil;

    int n = tp.get_variables().size();
    var_names_.clear();
    fact_names_.clear();
    var_names_.reserve(n);
    fact_names_.resize(n);

    for (int v = 0; v < n; ++v) {
        VariableProxy var = tp.get_variables()[v];
        var_names_.emplace_back(py::str(var.get_name()));

        int dom = var.get_domain_size();
        fact_names_[v].reserve(dom);
        for (int d = 0; d < dom; ++d)
            fact_names_[v].emplace_back(py::str(var.get_fact(d).get_name()));
    }
    tables_ready_ = true;

    if (debug_)
        utils::g_log << "[AntPlan] String tables built for "
                     << n << " variables." << endl;
}


// ===================================================================
//  Float -> non-negative int conversion
// ===================================================================

int AntPlanHeuristic::float_to_heuristic(double raw) const {
    /*
     * Formula:  h = clamp( round(raw * scale_factor) + offset, 0, MAX_H )
     *
     * Example (scale=10000, offset=100000):
     *   raw = -0.4130  ->  round(-4130) + 100000 = 95870
     *   raw = -0.0884  ->  round(-884)  + 100000 = 99116
     *
     * Lower (more negative) raw values produce lower h, which FD treats
     * as "closer to the goal" — i.e. preferred by the search.
     */
    if (isnan(raw) || isinf(raw))
        return 0;

    long scaled = lround(raw * scale_factor_);
    long h      = scaled + static_cast<long>(offset_);

    if (h < 0)     return 0;
    if (h > MAX_H) return MAX_H;
    return static_cast<int>(h);
}


// ===================================================================
//  State hashing & cache
// ===================================================================

uint64_t AntPlanHeuristic::hash_state(const State &state) const {
    uint64_t h = 14695981039346656037ULL;  // FNV-1a offset basis
    int n = task_proxy.get_variables().size();
    for (int v = 0; v < n; ++v) {
        uint64_t val = static_cast<uint64_t>(state[v].get_value());
        h = fnv1a_update(h, val + 0x9e3779b97f4a7c15ULL
                            + (static_cast<uint64_t>(v) << 1));
    }
    return h;
}

void AntPlanHeuristic::maybe_evict_cache() {
    if (cache_.size() > cache_max_entries_)
        cache_.clear();
}


// ===================================================================
//  Preferred operators (unchanged logic from original)
// ===================================================================

void AntPlanHeuristic::mark_preferred_operators_and_relaxed_plan(
        const State &state, PropID goal_id) {
    Proposition *goal = get_proposition(goal_id);
    if (goal->marked)
        return;

    goal->marked = true;
    OpID op_id = goal->reached_by;
    if (op_id == NO_OP)
        return;

    UnaryOperator *unary_op = get_operator(op_id);
    bool is_preferred = true;
    for (PropID precond : get_preconditions(op_id)) {
        mark_preferred_operators_and_relaxed_plan(state, precond);
        if (get_proposition(precond)->reached_by != NO_OP)
            is_preferred = false;
    }

    int op_no = unary_op->operator_no;
    if (op_no != -1) {
        relaxed_plan_[op_no] = true;
        if (is_preferred) {
            OperatorProxy op = task_proxy.get_operators()[op_no];
            assert(task_properties::is_applicable(op, state));
            set_preferred(op);
        }
    }
}


// ===================================================================
//  Main heuristic computation
// ===================================================================

int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    ++stat_calls_;

    State state = convert_ancestor_state(ancestor_state);

    // ---- cache lookup ----
    uint64_t key = 0;
    if (use_cache_) {
        key = hash_state(state);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            ++stat_cache_hits_;
            return it->second;
        }
        ++stat_cache_miss_;
    }

    if (!py_ready_)
        throw runtime_error("[AntPlan] Python not ready in compute_heuristic.");

    // ---- call Python ----
    int h = 0;
    try {
        py::gil_scoped_acquire gil;
        ensure_tables(task_proxy);

        int n = task_proxy.get_variables().size();
        py::dict d;
        for (int v = 0; v < n; ++v)
            d[var_names_[v]] = fact_names_[v][state[v].get_value()];

        double raw = py_to_double(py_cost_fn_(d));
        h = float_to_heuristic(raw);

        if (log_states_) {
            utils::g_log << "[AntPlan] raw=" << raw
                         << "  h=" << h << "\n";
            for (int v = 0; v < n; ++v)
                utils::g_log << "  " << task_proxy.get_variables()[v].get_name()
                             << " = " << state[v].get_name() << "\n";
        }

    } catch (const exception &e) {
        if (debug_) {
            try {
                py::gil_scoped_acquire gil2;
                string tb = py::cast<string>(
                    py::module::import("traceback").attr("format_exc")());
                utils::g_log << "[AntPlan] Python error: " << e.what()
                             << "\n" << tb << endl;
            } catch (...) {
                utils::g_log << "[AntPlan] Python error: "
                             << e.what() << endl;
            }
        }
        h = 0;
    }

    // ---- cache store ----
    if (use_cache_) {
        maybe_evict_cache();
        cache_[key] = h;
    }

    return h;
}


// ===================================================================
//  Plugin registration
// ===================================================================

static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Evaluates states via a Python cost function.  Raw float values "
        "are converted to non-negative integers with:  "
        "h = clamp(round(raw * scale) + offset, 0, 100000000).");

    // ---- Python binding ----
    parser.add_option<string>(
        "function",
        "Python function name (attribute in module).",
        "anticipatory_cost_fn");
    parser.add_option<string>(
        "module",
        "Python module to import (dotted path).",
        "antplan.scripts.eval_antplan_gripper");

    // ---- float -> int conversion ----
    parser.add_option<int>(
        "scale",
        "Multiplier applied to the raw Python float before rounding.  "
        "Larger values preserve more decimal precision.",
        "1000");
    parser.add_option<int>(
        "offset",
        "Constant added after scaling to shift negative values into the "
        "non-negative range.  Choose a value larger than the expected "
        "|min_cost| * scale.",
        "100000");

    // ---- performance / debug ----
    parser.add_option<bool>(
        "debug",
        "Print diagnostics and final statistics.",
        "false");
    parser.add_option<bool>(
        "log_states",
        "Log every state and its raw/converted h value (very slow).",
        "false");
    parser.add_option<bool>(
        "cache",
        "Memoize heuristic values by state hash.",
        "true");
    parser.add_option<int>(
        "cache_max_entries",
        "Max cache entries before full eviction.",
        "500000");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;

    return make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

}  // namespace antplan_heuristic