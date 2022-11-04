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

import React from 'react';
import Editor from '@monaco-editor/react';
import { SVGIDMapping, SVGOptions } from './types';
import { props_defaults } from 'examples';

import { HorizontalGroup, Label, Tooltip, Button, Input, VerticalGroup, stylesFactory } from '@grafana/ui';
import { PanelOptionsEditorProps, GrafanaTheme, PanelOptionsEditorBuilder } from '@grafana/data';
import { config } from '@grafana/runtime';

import { css } from 'emotion';

interface MonacoEditorProps {
  value: string;
  theme: string;
  language: string;
  onChange: (value?: string | undefined) => void;
}
class MonacoEditor extends React.PureComponent<MonacoEditorProps> {
  getEditorValue: any | undefined;
  editorInstance: any | undefined;

  // onSourceChange = () => {
  //   console.log(this.getEditorValue);
  //   this.props.onChange(this.getEditorValue());
  // };
  handleEditorChange = (getEditorValue: any, editorInstance: any) => {
    this.props.onChange(getEditorValue);
    console.log('here is the current model value:', getEditorValue);
  };
  onEditorDidMount = (editorInstance: any, getEditorValue: any) => {
    this.editorInstance = editorInstance;
    this.getEditorValue = getEditorValue;
  };
  updateDimensions() {
    this.editorInstance.layout();
  }
  render() {
    const source = this.props.value;
    if (this.editorInstance) {
      this.editorInstance.layout();
    }
    return (
      //<div onBlur={this.onSourceChange}>
      <Editor
        height={'33vh'}
        language={this.props.language}
        theme={this.props.theme}
        value={source}
        onMount={this.onEditorDidMount}
        onChange={this.handleEditorChange}
      />
      //</div>
    );
  }
}

interface SVGIDMappingProps {
  value: SVGIDMapping;
  index?: number;
  styles?: any;
  onChangeItem?: (a: SVGIDMapping, b: number) => void | undefined;
  onAdd?: (a: SVGIDMapping) => void;
  onDelete?: (a: number) => void;
}
class SvgMapping extends React.PureComponent<SVGIDMappingProps> {
  constructor(props: SVGIDMappingProps) {
    super(props);
    this.state = { ...props.value };
  }
  render() {
    const { value, index, onChangeItem, onAdd, onDelete } = this.props;
    return (
      <HorizontalGroup>
        <Label>SVG ID</Label>
        <Input
          type="text"
          name="svgId"
          defaultValue={value.svgId}
          onBlur={(e) => {
            const svgId = e.currentTarget.value;
            this.setState({ svgId: svgId });
            onChangeItem && index && onChangeItem({ ...value, svgId: svgId }, index);
          }}
        />
        <Label>Mapped Name</Label>
        <Input
          type="text"
          name="mappedName"
          defaultValue={value.mappedName}
          onBlur={(e) => {
            const mappedName = e.currentTarget.value;
            this.setState({ mappedName: mappedName });
            onChangeItem && index && onChangeItem({ ...value, mappedName: mappedName }, index);
          }}
        />
        {value.svgId && onDelete && index !== undefined && (
          <Tooltip content="Delete this mapping" theme={'info'}>
            <Button
              variant="destructive"
              icon="trash-alt"
              size="sm"
              onClick={() => {
                onDelete(index);
              }}
            >
              Remove
            </Button>
          </Tooltip>
        )}
        {!value.svgId && onAdd && (
          <Tooltip content="Add a new SVG Element ID to svgmap property mapping manually" theme={'info'}>
            <Button
              variant="secondary"
              size="sm"
              icon="plus-circle"
              onClick={() => {
                onAdd(this.state as SVGIDMapping);
              }}
            >
              Add
            </Button>
          </Tooltip>
        )}
      </HorizontalGroup>
    );
  }
}

class SvgMappings extends React.PureComponent<PanelOptionsEditorProps<SVGIDMapping[]>> {
  onChangeItem = (updatedMapping: SVGIDMapping, index: number) => {
    let newMappings = [...this.props.value];
    newMappings[index] = updatedMapping;
    this.props.onChange(newMappings);
  };
  onAdd = (newMapping: SVGIDMapping) => {
    if (newMapping.svgId !== '') {
      let newMappings = [...this.props.value, newMapping];
      this.props.onChange(newMappings);
    }
  };
  onDelete = (index: number) => {
    let newMappings = [...this.props.value];
    newMappings.splice(index, 1);
    this.props.onChange(newMappings);
  };
  render() {
    const styles = getStyles(config.theme);
    const svgMappings = this.props.value;
    return (
      <VerticalGroup>
        <HorizontalGroup>
          <Tooltip content="Clear all SVG Element ID to svgmap property mappings" theme="info">
            <Button
              variant="destructive"
              icon="trash-alt"
              size="sm"
              onClick={() => {
                this.props.onChange([]);
              }}
            >
              Clear All
            </Button>
          </Tooltip>
          <SvgMapping value={{ svgId: '', mappedName: '' }} styles={styles} onAdd={this.onAdd} />
        </HorizontalGroup>
        {svgMappings.map((currentMapping: SVGIDMapping, index: number) => {
          return (
            <SvgMapping
              key={currentMapping.svgId}
              value={currentMapping}
              index={index}
              onChangeItem={this.onChangeItem}
              onDelete={this.onDelete}
              styles={styles}
            />
          );
        })}
      </VerticalGroup>
    );
  }
}

// ---------------------------------------

export const optionsBuilder = (builder: PanelOptionsEditorBuilder<SVGOptions>) => {
  return builder
    .addBooleanSwitch({
      category: ['SVG Document'],
      path: 'svgAutoComplete',
      name: 'Enable SVG AutoComplete',
      description: 'Enable editor autocompletion, optional as it can be buggy on large documents',
    })
    .addCustomEditor({
      category: ['SVG Document'],
      path: 'svgSource',
      name: 'SVG Document',
      description: `Editor for SVG Document, while small tweaks can be made here, we recommend using a dedicated 
        Graphical SVG Editor and simply pasting the resulting XML here`,
      id: 'svgSource',
      defaultValue: props_defaults.svgNode,
      //editor: (props) => {
      editor: function myNamedFuntion(props) {
        const grafanaTheme = config.theme.name;
        return (
          <MonacoEditor
            language="xml"
            theme={grafanaTheme === 'Grafana Light' ? 'vs-light' : 'vs-dark'}
            value={props.value}
            onChange={props.onChange}
          />
        );
      },
    })
    .addBooleanSwitch({
      category: ['User JS Render'],
      path: 'eventAutoComplete',
      name: 'Enable Render JS AutoComplete',
      description: 'Enable editor autocompletion, optional as it can be buggy on large documents',
      defaultValue: true,
    })
    .addCustomEditor({
      category: ['User JS Render'],
      path: 'eventSource',
      name: 'User JS Render Code',
      description: `The User JS Render code is executed whenever new data is available, the root svg document is available as 'svgnode',
        and elements you've mapped using the SVG Mapping tools below are available as properties on the 'svgmap' object.
        The Grafana DataFrame is provided as 'data' and the 'options' object can be used to pass values and references between
        the Render context and the Init context`,
      id: 'eventSource',
      defaultValue: props_defaults.eventSource,
      //editor: (props) => {
      editor: function myNamedFunc(props) {
        const grafanaTheme = config.theme.name;
        return (
          <MonacoEditor
            language="javascript"
            theme={grafanaTheme === 'Grafana Light' ? 'vs-light' : 'vs-dark'}
            value={props.value}
            onChange={props.onChange}
          />
        );
      },
    })
    .addBooleanSwitch({
      category: ['User JS Init'],
      path: 'initAutoComplete',
      name: 'Enable Init JS AutoComplete',
      description: 'Enable editor autocompletion, optional as it can be buggy on large documents',
      defaultValue: true,
    })
    .addCustomEditor({
      category: ['User JS Init'],
      path: 'initSource',
      name: 'User JS Init Code',
      description: `The User JS Init code is executed once when the panel loads, you can use this to define helper functions that 
        you later reference in the User JS Render code section. The sections have identical execution contexts, and any 
        JS objects you want to reference between them will need to be attached to the options object as properties`,
      id: 'initSource',
      defaultValue: props_defaults.initSource,
      //editor: (props) => {
      editor: function myNamedFunc(props) {
        const grafanaTheme = config.theme.name;
        return (
          <MonacoEditor
            language="javascript"
            theme={grafanaTheme === 'Grafana Light' ? 'vs-light' : 'vs-dark'}
            value={props.value}
            onChange={props.onChange}
          />
        );
      },
    })
    .addBooleanSwitch({
      category: ['SVG Mapping'],
      path: 'addAllIDs',
      name: 'Add all SVG Element IDs',
      description:
        'Parse the SVG Document for Elements with IDs assigned and automatically add them to the mapping list',
      defaultValue: false,
    })
    .addBooleanSwitch({
      category: ['SVG Mapping'],
      path: 'captureMappings',
      name: 'Enable SVG Mapping on Click',
      description:
        'When activated, clicking an element in the panel will attempt to map the clicked element or its nearest parent element with an ID assigned',
      defaultValue: false,
    })
    .addCustomEditor({
      category: ['SVG Mapping'],
      id: 'svgMappings',
      path: 'svgMappings',
      name: 'SVG Mappings',
      description:
        'The SVG ID should match an element in the SVG document with an existing ID tag, the element will be attached to the "svgmap" object in the user code execution contexts as a property using the Mapped Name provided below',
      defaultValue: props_defaults.svgMappings,
      editor: SvgMappings,
    });
};

const getStyles = stylesFactory((theme: GrafanaTheme) => {
  return {
    colorPicker: css`
      padding: 0 ${theme.spacing.sm};
    `,
    inputPrefix: css`
      display: flex;
      align-items: center;
    `,
    trashIcon: css`
      color: ${theme.colors.textWeak};
      cursor: pointer;
      //
      &:hover {
        color: ${theme.colors.text};
      }
    `,
    addIcon: css`
      color: ${theme.colors.textWeak};
      cursor: pointer;
      //
      &:hover {
        color: ${theme.colors.text};
      }
    `,
  };
});
