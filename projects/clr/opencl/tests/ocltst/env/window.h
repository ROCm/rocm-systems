/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WINDOW_H_
#define _WINDOW_H_

#ifdef _WIN32

#include <gl\gl.h>
#include <windows.h>

class Window {
 public:
  typedef LRESULT (*WindowProc)(HWND hW, UINT uMsg, WPARAM wP, LPARAM lP);

 public:
  Window(const char* title, int x, int y, int width, int height, unsigned int uiStyle);
  ~Window();

  void ConsumeEvents(void);
  void ShowImage(unsigned int width, unsigned int height, float* data);

 private:
  static LRESULT WINAPI DefWindowProc(HWND hW, UINT uMsg, WPARAM wP, LPARAM lP);

  static void OnPaint(void);

 public:
  static HWND _hWnd;
  static unsigned char* _data;
  static unsigned int _w;
  static unsigned int _h;
};

#endif  // _WIN32

#endif  // _WINDOW_H_
