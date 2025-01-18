#ifndef CATAMARI_CONFIG_HH
#define CATAMARI_CONFIG_HH

// Control various optimizations/tradeoffs...
#define LOAD_MATRIX_OUTSIDE 1

// On-the-fly construction of the Schur complement buffers
// (setting to 1 obtains the original implementation, which is slightly slower for small matrices but saves memory).
#define ALLOCATE_SCHUR_COMPLEMENT_OTF 1

#define CATAMARI_FINEGRAINED_TIMERS 1

#if CATAMARI_FINEGRAINED_TIMERS
#define FG_START_TIMER(timer_struct, supernode, name) timer_struct(supernode, FineGrainedTimers::name).Start()
#define  FG_STOP_TIMER(timer_struct, supernode, name) timer_struct(supernode, FineGrainedTimers::name).Stop()
#else
#define FG_START_TIMER(timer_struct, supernode, name)
#define  FG_STOP_TIMER(timer_struct, supernode, name)
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#endif /* end of include guard: CATAMARI_CONFIG_HH */
