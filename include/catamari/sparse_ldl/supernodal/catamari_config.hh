#ifndef CATAMARI_CONFIG_HH
#define CATAMARI_CONFIG_HH

#define CATAMARI_FINEGRAINED_TIMERS 0
// #define CATAMARI_ENABLE_TIMERS

#if CATAMARI_FINEGRAINED_TIMERS
#define FG_START_TIMER(timer_struct, supernode, name) timer_struct(supernode, FineGrainedTimers::name).Start()
#define  FG_STOP_TIMER(timer_struct, supernode, name) timer_struct(supernode, FineGrainedTimers::name).Stop()
#else
#define FG_START_TIMER(timer_struct, supernode, name)
#define  FG_STOP_TIMER(timer_struct, supernode, name)
#endif  // ifdef CATAMARI_FINEGRAINED_TIMERS

#endif /* end of include guard: CATAMARI_CONFIG_HH */
