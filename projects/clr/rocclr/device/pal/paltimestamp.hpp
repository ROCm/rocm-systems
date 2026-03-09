/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/pal/paldefs.hpp"
#include "device/pal/palresource.hpp"

/*! \addtogroup pal PAL Resource Implementation
 *  @{
 */

//! PAL Device Implementation
namespace amd::pal {

class Device;
class VirtualGPU;
class Memory;

class TimeStamp : public amd::HeapObject {
 public:
  //! Enums for the timestamp information
  //! \note *4 is the limitaiton of SDMA HW
  //! (address has to be aligned by 256 bit)
  enum TimeStampValue { CommandStartTime = 0, CommandEndTime = 4, CommandTotal = 8 };

  //! The TimeStamp object flags
  union Flags {
    struct {
      uint32_t beginIssued_ : 1;
      uint32_t endIssued_ : 1;
    };
    uint32_t value_;
    Flags() : value_(0) {}
  };

  //! Default constructor
  TimeStamp(const VirtualGPU& gpu,  //!< Virtual GPU
            Pal::IGpuMemory* iMem,  //!< Buffer with the timer values
            uint memOffset,         //!< Offset in the buffer for the current TS
            address cpuAddr         //!< CPU pointer for the values in memory
  );

  //! Default destructor
  ~TimeStamp();

  //! Starts the timestamp
  void begin();

  //! Ends the timestamp
  void end();

  //! Returns the timestamp result in nano seconds
  void value(uint64_t* startTime, uint64_t* endTime);

  //! Clear all TimeStamp states
  void clearStates() {
    flags_.value_ = 0;
    values_[CommandStartTime] = 0;
    values_[CommandEndTime] = 0;
  }

  //! Timer commands were submitted to HW
  bool isValid() const { return (flags_.endIssued_) ? true : false; }

 private:
  //! Disable copy constructor
  TimeStamp(const TimeStamp&);

  //! Disable operator=
  TimeStamp& operator=(const TimeStamp&);

  //! Returns the GPU device object
  const VirtualGPU& gpu() const { return gpu_; }

  const VirtualGPU& gpu_;      //!< Virtual GPU
  Flags flags_;                //!< The time stamp state
  Pal::IGpuMemory* iMem_;      //!< Buffer with the timer values
  uint memOffset_;             //!< Offset in the buffer for the current timer
  volatile uint64_t* values_;  //!< CPU pointer to the timer values
};

class TimeStampCache : public amd::HeapObject {
 public:
  //! Default constructor
  TimeStampCache(VirtualGPU& gpu  //!< Virtual GPU object
                 )
      : gpu_(gpu), tsBufCpu_(NULL), tsOffset_(0) {}

  //! Default destructor
  ~TimeStampCache();

  //! Gets a time stamp object. It will find a freed object or allocate a new one
  TimeStamp* allocTimeStamp();

  //! Frees a time stamp object
  void freeTimeStamp(TimeStamp* ts) { freedTS_.push_back(ts); }

 private:
  static constexpr uint TimerSlotSize = TimeStamp::CommandTotal * sizeof(uint64_t);
  static constexpr uint TimerBufSize = TimerSlotSize * 4096;

  //! Disable copy constructor
  TimeStampCache(const TimeStampCache&);

  //! Disable operator=
  TimeStampCache& operator=(const TimeStampCache&);

  std::vector<TimeStamp*> freedTS_;  //!< Array of freed time stamp objects
  VirtualGPU& gpu_;                  //!< Virtual GPU
  std::vector<Memory*> tsBuf_;       //!< Array of memory objects with the timer value
  address tsBufCpu_;                 //!< CPU pointer for current TS memory
  uint tsOffset_;                    //!< Active offset in the current mem object
};

/*@}*/  // namespace amd::pal
}  // namespace amd::pal
