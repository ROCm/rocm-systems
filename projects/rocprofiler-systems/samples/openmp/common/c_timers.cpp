/*
MIT License

Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
    GMAP website: https://gmap.pucrs.br

    Pontifical Catholic University of Rio Grande do Sul (PUCRS)
    Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

The original NPB 3.4.1 version was written in Fortran and belongs to:
    http://www.nas.nasa.gov/Software/NPB/

------------------------------------------------------------------------------

The serial C++ version is a translation of the original NPB 3.4.1
Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER

Authors of the C++ code:
    Dalvan Griebler <dalvangriebler@gmail.com>
    Gabriell Araujo <hexenoften@gmail.com>
    Júnior Löff <loffjh@gmail.com>
*/

#include "wtime.hpp"
#include <cstdlib>

/*  prototype  */
void
wtime(double*);

/*****************************************************************/
/******         E  L  A  P  S  E  D  _  T  I  M  E          ******/
/*****************************************************************/
double
elapsed_time(void)
{
    double t;
    wtime(&t);
    return (t);
}

double start[64], elapsed[64];

/*****************************************************************/
/******            T  I  M  E  R  _  C  L  E  A  R          ******/
/*****************************************************************/
void
timer_clear(int n)
{
    elapsed[n] = 0.0;
}

/*****************************************************************/
/******            T  I  M  E  R  _  S  T  A  R  T          ******/
/*****************************************************************/
void
timer_start(int n)
{
    start[n] = elapsed_time();
}

/*****************************************************************/
/******            T  I  M  E  R  _  S  T  O  P             ******/
/*****************************************************************/
void
timer_stop(int n)
{
    double t, now;
    now = elapsed_time();
    t   = now - start[n];
    elapsed[n] += t;
}

/*****************************************************************/
/******            T  I  M  E  R  _  R  E  A  D             ******/
/*****************************************************************/
double
timer_read(int n)
{
    return (elapsed[n]);
}
