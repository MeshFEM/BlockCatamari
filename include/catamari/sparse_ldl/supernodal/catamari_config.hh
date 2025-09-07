#ifndef CATAMARI_CONFIG_HH
#define CATAMARI_CONFIG_HH

#define CATAMARI_FINEGRAINED_TIMERS 0
#define CATAMARI_FILLZERO_LOWER_TRI_ONLY 1 // Seems much faster on x86; a bit slower on Apple Silicon
#define PARALLELIZE_MERGE_INTO_SUPERNODE 1
// #define CATAMARI_ENABLE_TIMERS

#if CATAMARI_FINEGRAINED_TIMERS
#define FG_START_TIMER(timer_struct, supernode, name) timer_struct(supernode, decltype(timer_struct)::name).Start()
#define  FG_STOP_TIMER(timer_struct, supernode, name) timer_struct(supernode, decltype(timer_struct)::name).Stop()
#else
#define FG_START_TIMER(timer_struct, supernode, name)
#define  FG_STOP_TIMER(timer_struct, supernode, name)
#endif  // ifdef CATAMARI_FINEGRAINED_TIMERS

#endif /* end of include guard: CATAMARI_CONFIG_HH */
