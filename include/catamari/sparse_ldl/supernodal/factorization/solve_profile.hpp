#ifndef CATAMARI_SOLVE_PROFILE_HPP
#define CATAMARI_SOLVE_PROFILE_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <vector>

namespace catamari { namespace supernodal_ldl {

// Optional operation timelines, using the same categories as the existing
// fine-grained solve timers. Mutate/export only between solves. A supernode is
// owned by one task at a time; nested dense tasks are included in its elapsed
// interval, not counted again as worker CPU time. Concurrent solves on the
// same factor are unsupported, as with the solver's other mutable scratch.
struct SolveProfile {
    using Clock = std::chrono::steady_clock;
    using Type = FineGrainedTimersSolve::Type;
    struct Event { int64_t begin, end; Int solve, operation, thread; };
    struct Node {
        bool selected = false;
        std::array<int64_t, FineGrainedTimersSolve::NumTimers> starts{};
        std::array<Int, FineGrainedTimersSolve::NumTimers> threads{};
        std::vector<Event> events;
    };
    bool enabled = false;
    Int solve = -1;
    int64_t origin = 0;
    std::vector<Node> nodes;
    Node phases;

    static int64_t now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    }
    void configure(const SymmetricOrdering &ordering, bool on, int max_depth, Int min_width) {
        enabled = false;
        nodes.clear();
        phases = Node{};
        if (!on) return;
        nodes.resize(ordering.supernode_sizes.Size());
        std::vector<std::pair<Int, int>> stack;
        for (Int r : ordering.assembly_forest.roots) stack.emplace_back(r, 0);
        while (!stack.empty()) {
            const auto [s, depth] = stack.back(); stack.pop_back();
            nodes[s].selected = (max_depth < 0 && min_width == 0) ||
                (max_depth >= 0 && depth <= max_depth) ||
                (min_width > 0 && ordering.supernode_sizes[s] >= min_width);
            if (nodes[s].selected) nodes[s].events.reserve(128);
            const auto &af = ordering.assembly_forest;
            for (Int i = af.child_offsets[s]; i < af.child_offsets[s + 1]; ++i)
                stack.emplace_back(af.children[i], depth + 1);
        }
        phases.selected = true;
        phases.events.reserve(256);
        enabled = true;
        reset();
    }
    void reset() {
        for (auto &n : nodes) n.events.clear(); // Retain capacity for warm traces.
        phases.events.clear();
        solve = -1;
        origin = now();
    }
    void start(Int s, Type op) {
        if (!enabled) return;
        auto &n = s < 0 ? phases : nodes[s];
        if (!n.selected) return;
        n.threads[op] = tbb::this_task_arena::current_thread_index();
        n.starts[op] = now();
    }
    void stop(Int s, Type op) {
        if (!enabled) return;
        auto &n = s < 0 ? phases : nodes[s];
        if (!n.selected) return;
        const auto end = now();
        n.events.push_back({n.starts[op] - origin, end - origin, solve, Int(op), n.threads[op]});
    }
    struct Scope {
        SolveProfile &p; Int s; Type op;
        Scope(SolveProfile &p_, Int s_, Type op_) : p(p_), s(s_), op(op_) { p.start(s, op); }
        ~Scope() { p.stop(s, op); }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
    };
    void write(const std::string &path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot open solve profile: " + path);
        out << "solve,node,operation,thread,begin_ns,end_ns\n";
        auto emit = [&](Int s, const Node &n) {
            for (const auto &e : n.events)
                out << e.solve << ',' << s << ',' << FineGrainedTimersSolve::nameForType(Type(e.operation))
                    << ',' << e.thread << ',' << e.begin << ',' << e.end << '\n';
        };
        emit(-1, phases);
        for (size_t s = 0; s < nodes.size(); ++s) emit(Int(s), nodes[s]);
        if (!out) throw std::runtime_error("Cannot write solve profile: " + path);
    }
};

}}

// Existing aggregate timers remain compile-time optional. The timeline is
// independently enabled at runtime; both use the same operation names/hooks.
#define SOLVE_START_TIMER(s, name) do { \
    FG_START_TIMER(solve_shared_state_.finegrained_timers, s, name); \
    solve_profile_.start(s, FineGrainedTimersSolve::name); \
} while (false)
#define SOLVE_STOP_TIMER(s, name) do { \
    solve_profile_.stop(s, FineGrainedTimersSolve::name); \
    FG_STOP_TIMER(solve_shared_state_.finegrained_timers, s, name); \
} while (false)

#endif
