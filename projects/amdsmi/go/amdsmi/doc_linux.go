// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

// Package amdsmi provides read-only Linux GPU queries through AMD SMI 27.1 and CGO.
// Calls are serialized. Balance Init with ShutDown and rediscover handles after final shutdown.
package amdsmi
