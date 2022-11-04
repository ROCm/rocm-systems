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

import React, { PureComponent } from 'react';

import { PanelProps } from '@grafana/data';
import { stylesFactory } from '@grafana/ui';
import { SVG, Element as SVGElement, Dom as SVGDom, extend as SVGExtend, Runner as SVGRunner } from '@svgdotjs/svg.js';
import { SVGOptions, SVGIDMapping } from 'types';

import { css } from 'emotion';

interface MappedElements {
  [key: string]: SVGElement | SVGDom;
}
interface Props extends PanelProps<SVGOptions> {}
interface PanelState {
  addAllIDs: boolean;
  svgNode: SVGElement | SVGDom | null;
  svgSource: string | null;
  mappedElements: MappedElements | null;
  svgMappings: SVGIDMapping[];
  initFunctionSource: string;
  initFunction: Function | null;
  eventFunctionSource: string;
  eventFunction: Function | null;
  initialized: boolean;
}
interface TextMappedElement extends SVGElement {
  textElement: Element;
}

SVGExtend(SVGElement, {
  openOnClick: function (this: SVGElement, url: string) {
    return window.open(url);
  },
  animateContRotate: function (this: SVGElement, speed: number) {
    return (
      this.animate(speed)
        //@ts-ignore
        .ease('-')
        //@ts-ignore
        .rotate(360)
        .loop()
    );
  },
  showOn: function (this: SVGElement, on: boolean) {
    if (on) {
      this.show();
    } else {
      this.hide();
    }
  },
  animateOn: function (this: SVGElement, speed: number, on: boolean, animation: Function) {
    if (on) {
      //@ts-ignore
      if (this.timeline()._runners.length === 0) {
        animation(this.animate(speed));
      } else {
        this.timeline().play();
      }
    } else {
      this.timeline().stop();
    }
  },
  stopAnimation: function (this: SVGRunner) {
    this.timeline().stop();
  },
  getParentNode: function (this: SVGElement) {
    return this.node.parentNode;
  },
  getTopNode: function (this: SVGElement) {
    let currentNode: Element = this.node as Element;
    while (true) {
      if (currentNode.parentNode && !currentNode.className.includes('svg-object')) {
        currentNode = currentNode.parentNode as Element;
      } else {
        return currentNode;
      }
    }
  },
});
SVGExtend(SVGDom, {
  updateXHTMLFontText: function (this: SVGDom, newText: string) {
    let currentElement: Element | TextMappedElement = this.node;
    let i = 0;
    while (currentElement.localName !== 'xhtml:font') {
      if (currentElement.firstElementChild && i < 10) {
        currentElement = currentElement.firstElementChild;
        i++;
      } else {
        return;
      }
    }
    currentElement.innerHTML = newText;
  },
});

export class SVGPanel extends PureComponent<Props, PanelState> {
  constructor(props: any) {
    super(props);
    this.state = {
      addAllIDs: false,
      svgNode: null,
      svgSource: null,
      svgMappings: [],
      mappedElements: null,
      initFunctionSource: '',
      initFunction: null,
      eventFunctionSource: '',
      eventFunction: null,
      initialized: false,
    };
  }

  initializeMappings(svgNode: SVGElement | SVGDom) {
    const svgMappings = this.props.options.svgMappings;
    let currentElements: MappedElements = { ...this.state.mappedElements };
    currentElements = {};
    console.log('Current length is ', svgMappings.length);
    for (let i = 0; i < svgMappings.length; i++) {
      if (svgMappings[i].mappedName !== '') {
        currentElements[this.props.options.svgMappings[i].mappedName] = svgNode.findOne(
          `#${this.props.options.svgMappings[i].svgId}`
        )!;
      }
    }
    this.setState({ mappedElements: currentElements });
  }
  mapAllIDs(svgNode: SVGDom) {
    let svgMappings: SVGIDMapping[] = [...this.props.options.svgMappings];
    let nodeFilterID: NodeFilter = {
      acceptNode: (node: Element) => {
        if (node.id) {
          if (node.id !== '') {
            return NodeFilter.FILTER_ACCEPT;
          }
        }
        return NodeFilter.FILTER_REJECT;
      },
    };
    let svgWalker = document.createTreeWalker(svgNode.node, NodeFilter.SHOW_ALL, nodeFilterID);
    let currentNode: Element | null = svgWalker.currentNode as Element;
    while (currentNode) {
      if (currentNode && currentNode.id) {
        if (svgMappings.filter((mapping) => (currentNode ? mapping.svgId === currentNode.id : false)).length === 0) {
          svgMappings.push({ svgId: currentNode.id, mappedName: '' });
        }
      }
      currentNode = svgWalker.nextNode() as Element;
    }
    this.setState({ svgMappings: [...svgMappings], initialized: false });
    this.props.options.svgMappings = [...svgMappings];
    this.props.onOptionsChange(this.props.options);
    this.forceUpdate();
  }

  mappingClickHandler(event: React.MouseEvent<HTMLElement, MouseEvent>) {
    if (event.target) {
      let clicked = event.target as Element;
      let loopCount = 0;
      let svgMappings: SVGIDMapping[] = [...this.props.options.svgMappings];
      if (clicked.id) {
        while (clicked.id === '') {
          loopCount++;
          if (loopCount > 20) {
            return;
          }
          clicked = clicked.parentNode as Element;
        }
        for (let i = 0; i < svgMappings.length; i++) {
          if (svgMappings[i].svgId === clicked.id) {
            return;
          }
        }
        svgMappings.push({ svgId: clicked.id, mappedName: '' });
        this.setState({ svgMappings: [...svgMappings], initialized: false });
        this.props.options.svgMappings = [...svgMappings];
        this.props.onOptionsChange(this.props.options);
        this.forceUpdate();
      }
    }
  }
  renderSVG(element: SVGSVGElement | SVGDom | null) {
    if (element) {
      if (
        this.props.options.initSource !== this.state.initFunctionSource ||
        this.state.addAllIDs !== this.props.options.addAllIDs
      ) {
        this.setState({
          initFunctionSource: this.props.options.initSource,
          addAllIDs: this.props.options.addAllIDs,
          initialized: false,
        });
      }
      if (!this.state.initialized) {
        console.log('initializing');
        let svgNode = SVG(element);
        svgNode.clear();
        svgNode.svg(this.props.options.svgSource);
        svgNode.size(this.props.width, this.props.height);
        if (this.props.options.addAllIDs) {
          this.mapAllIDs(svgNode);
        }
        this.initializeMappings(svgNode);
        this.setState({ svgNode: svgNode });

        try {
          this.setState({
            initFunction: Function(
              'data',
              'options',
              'svgnode',
              'svgmap',
              this.props.replaceVariables(this.props.options.initSource)
            ),
          });
          if (this.state.mappedElements && this.state.initFunction) {
            this.state.initFunction(this.props.data, this.props.options, this.state.svgNode, this.state.mappedElements);
            this.setState({ initialized: true });
          }
        } catch (e) {
          this.setState({ initialized: true });
          console.log(`User init code failed: ${e}`);
        }
      }

      try {
        let eventFunction = this.state.eventFunction;
        if (this.props.options.eventSource !== this.state.eventFunctionSource) {
          let eventFunctionSource = this.props.options.eventSource;
          eventFunction = Function(
            'data',
            'options',
            'svgnode',
            'svgmap',
            this.props.replaceVariables(eventFunctionSource)
          );
          this.setState({ eventFunctionSource: eventFunctionSource, eventFunction: eventFunction, initialized: false });
        }
        if (this.state.mappedElements && eventFunction) {
          eventFunction(this.props.data, this.props.options, this.state.svgNode, this.state.mappedElements);
        }
      } catch (e) {
        console.log(`User event code failed: ${e}`);
      }

      return this.state.svgNode ? this.state.svgNode.svg() : null;
    } else {
      return null;
    }
  }
  render() {
    const styles = this.getStyles();
    return (
      <div
        className={styles.wrapper}
        onClick={this.props.options.captureMappings ? this.mappingClickHandler.bind(this) : undefined}
      >
        <svg
          style={{
            width: `${this.props.width}px`,
            height: `${this.props.height}px`,
          }}
          className={'svg-object'}
          ref={(ref) => this.renderSVG(ref)}
        ></svg>
      </div>
    );
  }

  getStyles = stylesFactory(() => {
    return {
      wrapper: css`
        position: relative;
      `,
      svg: css`
        position: absolute;
        top: 0;
        left: 0;
      `,
      textBox: css`
        position: absolute;
        bottom: 0;
        left: 0;
        padding: 10px;
      `,
    };
  });
}
