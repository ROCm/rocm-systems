
#ifndef _SRC_CORE_PROFILING_LOCK_H
#define _SRC_CORE_PROFILING_LOCK_H

enum LockMode{
  PROFILER_V1_LOCK,
  PROFILER_V2_LOCK,
};

class ProfilingLock {
public:
  static void Lock(LockMode mode);
  ~ProfilingLock();

private:
  ProfilingLock();
  static ProfilingLock *Instance();

  const char *lock_file;
  const char *pid_file;
  int lock;
};

#endif