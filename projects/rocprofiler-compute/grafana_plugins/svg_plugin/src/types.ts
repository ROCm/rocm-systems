// ################################################################################
// # Copyright (c) 2020 ACE IoT Solutions LLC
// #
// # Permission is hereby granted, free of charge, to any person obtaining a copy
// # of this software and associated documentation files (the "Software"), to deal
// # in the Software without restriction, including without limitation the rights
// # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// # copies of the Software, and to permit persons to whom the Software is
// # furnished to do so, subject to the following conditions:

// # The above copyright notice and this permission notice shall be included in all
// # copies or substantial portions of the Software.
// #
// # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// # SOFTWARE.
// ################################################################################

type SeriesSize = 'sm' | 'md' | 'lg';

export interface SimpleOptions {
  text: string;
  showSeriesCount: boolean;
  seriesCountSize: SeriesSize;
}
export interface SVGIDMapping {
  svgId: string;
  mappedName: string;
}
export interface SVGOptions {
  captureMappings: boolean;
  addAllIDs: boolean;
  svgSource: string;
  svgAutocomplete: boolean;
  eventSource: string;
  eventAutocomplete: boolean;
  initSource: string;
  initAutocomplete: boolean;
  svgMappings: SVGIDMapping[];
}
export interface SVGDefaults {
  svgNode: string; //svg default text
  initSource: string; //init default text
  eventSource: string; //render default text
  svgMappings: SVGIDMapping[]; //default mappings
}
