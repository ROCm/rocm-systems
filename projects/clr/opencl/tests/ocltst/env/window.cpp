/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifdef _WIN32

#include <assert.h>
#include <stdio.h>
#include <windows.h>

#include "Window.h"

HWND Window::_hWnd;
unsigned char* Window::_data;
unsigned int Window::_w;
unsigned int Window::_h;

void Window::OnPaint(void) {
  PAINTSTRUCT ps;
  HDC hDC = BeginPaint(_hWnd, &ps);

  if (_w && _h && _data) {
    BITMAPINFO bm;
    bm.bmiColors[0].rgbBlue = 0;
    bm.bmiColors[0].rgbGreen = 0;
    bm.bmiColors[0].rgbRed = 0;
    bm.bmiColors[0].rgbReserved = 0;

    bm.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bm.bmiHeader.biWidth = _w;
    bm.bmiHeader.biHeight = _h;
    bm.bmiHeader.biPlanes = 1;
    bm.bmiHeader.biBitCount = 32;
    bm.bmiHeader.biCompression = BI_RGB;
    bm.bmiHeader.biSizeImage = 0;
    bm.bmiHeader.biXPelsPerMeter = 0;
    bm.bmiHeader.biYPelsPerMeter = 0;
    bm.bmiHeader.biClrUsed = 0;
    bm.bmiHeader.biClrImportant = 0;

    int ret = SetDIBitsToDevice(hDC, 0, 0, _w, _h, 0, 0, 0, _h, _data, &bm, DIB_RGB_COLORS);
    assert(ret);
  }

  EndPaint(_hWnd, &ps);
}

/*****************************************************************************
 *****************************************************************************/
LRESULT WINAPI Window::DefWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
    case WM_CHAR:
      switch (wParam) {
        case 27:  // ESC
          exit(0);
          break;
      }
      return 0;
    case WM_PAINT:
      OnPaint();
      return 0;
  }
  return ::DefWindowProc(hWnd, uMsg, wParam, lParam);
}

Window::Window(const char* title, int x, int y, int width, int height, unsigned int uiStyle) {
  _data = NULL;
  _w = 0;
  _h = 0;

  WNDCLASS wc = {0,
                 (WNDPROC)Window::DefWindowProc,
                 0,
                 0,
                 GetModuleHandle(0),
                 LoadIcon(NULL, IDI_WINLOGO),
                 LoadCursor(NULL, IDC_ARROW),
                 NULL,
                 NULL,
                 "TST"};
  if (!RegisterClass(&wc)) {
    MessageBox(NULL, "RegisterClass() failed", "Error", MB_OK);
    exit(0);
  }

  if (uiStyle == 0) {
    uiStyle = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
  }

  RECT r = {x, y, x + width, y + height};
  AdjustWindowRect(&r, uiStyle, 0);

  _hWnd = CreateWindow("TST", title, uiStyle, r.left, r.top, r.right - r.left, r.bottom - r.top,
                       NULL, NULL, GetModuleHandle(0), this);
  if (_hWnd == NULL) {
    MessageBox(NULL, "CreateWindow() failed.", "Error", MB_OK);
    exit(0);
  }

  ShowWindow(_hWnd, SW_SHOW);
  UpdateWindow(_hWnd);
}

Window::~Window(void) {
  DestroyWindow(_hWnd);

  if (_data) {
    delete[] _data;
  }

  UnregisterClass("TST", GetModuleHandle(NULL));
}

void Window::ConsumeEvents(void) {
  while (1) {
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) {
      GetMessage(&msg, NULL, 0, 0);
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }
}

void Window::ShowImage(unsigned int width, unsigned int height, float* data) {
  if (_data) {
    delete[] _data;
  }

  _data = new unsigned char[4 * width * height];
  _w = width;
  _h = height;

  unsigned char* pb = _data;
  float* p = data;
  unsigned int i;
  for (i = 0; i < (unsigned int)(width * height); i++) {
    //
    //  argb
    //
    float v = p[2] > 1.f ? 1.f : (p[2] < 0.f ? 0.f : p[2]);
    *pb++ = (unsigned char)(255.f * v);
    v = p[1] > 1.f ? 1.f : (p[1] < 0.f ? 0.f : p[1]);
    *pb++ = (unsigned char)(255.f * v);
    v = p[0] > 1.f ? 1.f : (p[0] < 0.f ? 0.f : p[0]);
    *pb++ = (unsigned char)(255.f * v);
    v = p[3] > 1.f ? 1.f : (p[3] < 0.f ? 0.f : p[3]);
    *pb++ = (unsigned char)(255.f * v);
    p += 4;
  }

  RedrawWindow(_hWnd, NULL, NULL, RDW_INVALIDATE);
  OnPaint();
}

#endif  // _WIN32
