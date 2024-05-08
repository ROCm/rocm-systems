=====================
Python API reference
=====================

This section describes the ROCm SMI Python module API.

.. default-domain:: py
.. py:currentmodule:: rocm_smi

Functions
---------

.. autofunction:: rocm_smi.driverInitialized

.. autofunction:: rocm_smi.formatJson

.. autofunction:: rocm_smi.formatCsv

.. autofunction:: rocm_smi.formatMatrixToJSON

.. autofunction:: rocm_smi.getBus

.. autofunction:: rocm_smi.getFanSpeed

.. autofunction:: rocm_smi.getGpuUse

.. autofunction:: rocm_smi.getDRMDeviceId

.. autofunction:: rocm_smi.getSubsystemId

.. autofunction:: rocm_smi.getVendor

.. autofunction:: rocm_smi.getGUID

.. autofunction:: rocm_smi.getTargetGfxVersion

.. autofunction:: rocm_smi.getNodeId

.. autofunction:: rocm_smi.getDeviceName

.. autofunction:: rocm_smi.getRev

.. autofunction:: rocm_smi.getMaxPower

.. autofunction:: rocm_smi.getMemInfo

.. autofunction:: rocm_smi.getProcessName

.. autofunction:: rocm_smi.getPerfLevel

.. autofunction:: rocm_smi.getPid

.. autofunction:: rocm_smi.getPidList

.. autofunction:: rocm_smi.getPower

.. autofunction:: rocm_smi.getRasEnablement

.. autofunction:: rocm_smi.getTemp

.. autofunction:: rocm_smi.findFirstAvailableTemp

.. autofunction:: rocm_smi.getTemperatureLabel

.. autofunction:: rocm_smi.getPowerLabel

.. autofunction:: rocm_smi.getVbiosVersion

.. autofunction:: rocm_smi.getVersion

.. autofunction:: rocm_smi.getComputePartition

.. autofunction:: rocm_smi.getMemoryPartition

.. autofunction:: rocm_smi.print2DArray

.. autofunction:: rocm_smi.printEmptyLine

.. autofunction:: rocm_smi.printErrLog

.. autofunction:: rocm_smi.printInfoLog

.. autofunction:: rocm_smi.printEventList

.. autofunction:: rocm_smi.printLog

.. autofunction:: rocm_smi.printListLog

.. autofunction:: rocm_smi.printLogSpacer

.. autofunction:: rocm_smi.printSysLog

.. autofunction:: rocm_smi.printTableLog

.. autofunction:: rocm_smi.printTableRow

.. autofunction:: rocm_smi.checkIfSecondaryDie

.. autofunction:: rocm_smi.resetClocks

.. autofunction:: rocm_smi.resetFans

.. autofunction:: rocm_smi.resetPowerOverDrive

.. autofunction:: rocm_smi.resetProfile

.. autofunction:: rocm_smi.resetXgmiErr

.. autofunction:: rocm_smi.resetPerfDeterminism

.. autofunction:: rocm_smi.resetComputePartition

.. autofunction:: rocm_smi.resetMemoryPartition

.. autofunction:: rocm_smi.setClockRange

.. autofunction:: rocm_smi.setClockExtremum

.. autofunction:: rocm_smi.setVoltageCurve

.. autofunction:: rocm_smi.setPowerPlayTableLevel

.. autofunction:: rocm_smi.setClockOverDrive

.. autofunction:: rocm_smi.setClocks

.. autofunction:: rocm_smi.setPerfDeterminism

.. autofunction:: rocm_smi.resetGpu

.. autofunction:: rocm_smi.isRasControlAvailable

.. autofunction:: rocm_smi.setRas

.. autofunction:: rocm_smi.setFanSpeed

.. autofunction:: rocm_smi.setPerformanceLevel

.. autofunction:: rocm_smi.setPowerOverDrive

.. autofunction:: rocm_smi.setProfile

.. autofunction:: rocm_smi.setComputePartition

.. autofunction:: rocm_smi.progressbar

.. autofunction:: rocm_smi.showProgressbar

.. autofunction:: rocm_smi.setMemoryPartition

.. autofunction:: rocm_smi.showVersion

.. autofunction:: rocm_smi.showAllConcise

.. autofunction:: rocm_smi.showAllConciseHw

.. autofunction:: rocm_smi.showBus

.. autofunction:: rocm_smi.showClocks

.. autofunction:: rocm_smi.showCurrentClocks

.. autofunction:: rocm_smi.showCurrentFans

.. autofunction:: rocm_smi.showCurrentTemps

.. autofunction:: rocm_smi.showFwInfo

.. autofunction:: rocm_smi.showGpusByPid

.. autofunction:: rocm_smi.getCoarseGrainUtil

.. autofunction:: rocm_smi.showGpuUse

.. autofunction:: rocm_smi.showEnergy

.. autofunction:: rocm_smi.showId

.. autofunction:: rocm_smi.showMaxPower

.. autofunction:: rocm_smi.showMemInfo

.. autofunction:: rocm_smi.showMemUse

.. autofunction:: rocm_smi.showMemVendor

.. autofunction:: rocm_smi.showOverDrive

.. autofunction:: rocm_smi.showPcieBw

.. autofunction:: rocm_smi.showPcieReplayCount

.. autofunction:: rocm_smi.showPerformanceLevel

.. autofunction:: rocm_smi.showPids

.. autofunction:: rocm_smi.showPower

.. autofunction:: rocm_smi.showPowerPlayTable

.. autofunction:: rocm_smi.showProduct

.. autofunction:: rocm_smi.showProfile

.. autofunction:: rocm_smi.showRange

.. autofunction:: rocm_smi.showRasInfo

.. autofunction:: rocm_smi.showRetiredPages

.. autofunction:: rocm_smi.showSerialNumber

.. autofunction:: rocm_smi.showUId

.. autofunction:: rocm_smi.showVbiosVersion

.. autofunction:: rocm_smi.showEvents

.. autofunction:: rocm_smi.showDriverVersion

.. autofunction:: rocm_smi.showVoltage

.. autofunction:: rocm_smi.showVoltageCurve

.. autofunction:: rocm_smi.showXgmiErr

.. autofunction:: rocm_smi.showAccessibleTopology

.. autofunction:: rocm_smi.showWeightTopology

.. autofunction:: rocm_smi.showHopsTopology

.. autofunction:: rocm_smi.showTypeTopology

.. autofunction:: rocm_smi.showNumaTopology

.. autofunction:: rocm_smi.showHwTopology

.. autofunction:: rocm_smi.showNodesBw

.. autofunction:: rocm_smi.showComputePartition

.. autofunction:: rocm_smi.showMemoryPartition

.. autofunction:: rocm_smi.checkAmdGpus

.. autofunction:: rocm_smi.component_str

.. autofunction:: rocm_smi.confirmOutOfSpecWarning

.. autofunction:: rocm_smi.doesDeviceExist

.. autofunction:: rocm_smi.initializeRsmi

.. autofunction:: rocm_smi.isAmdDevice

.. autofunction:: rocm_smi.listDevices

.. autofunction:: rocm_smi.load

.. autofunction:: rocm_smi.padHexValue

.. autofunction:: rocm_smi.profileString

.. autofunction:: rocm_smi.relaunchAsSudo

.. autofunction:: rocm_smi.rsmi_ret_ok

.. autofunction:: rocm_smi.save
