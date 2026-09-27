latency:
  global_load_dword v0, v1, s[0:1]
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  s_waitcnt vmcnt(0)
  global_store_dword v1, v0, s[0:1]
  s_endpgm
