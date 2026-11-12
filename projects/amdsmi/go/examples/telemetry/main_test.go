// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestShutdownErrors(t *testing.T) {
	primary := errors.New("query failed")
	cleanup := errors.New("shutdown failed")
	for _, test := range []struct {
		name                   string
		primary, cleanup, want error
		message                string
	}{
		{"success", nil, nil, nil, ""},
		{"query", primary, nil, primary, ""},
		{"shutdown", nil, cleanup, cleanup, ""},
		{"both", primary, cleanup, primary, cleanup.Error()},
	} {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer
			got := test.primary
			recordShutdownError(&got, test.cleanup, &output)
			if !errors.Is(got, test.want) {
				t.Fatalf("want %v, got %v", test.want, got)
			}
			if strings.TrimSpace(output.String()) != test.message {
				t.Fatalf("want shutdown diagnostic %q, got %q", test.message, output.String())
			}
		})
	}
}
