#!/usr/bin/env python3

##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import ast
import astunparse
import regex
import os
import pandas as pd
import numpy as np


class CodeTransformer(ast.NodeTransformer):
    """
    Python AST visitor to transform user defined equation string to df format
    """

    def visit_Call(self, node):
        self.generic_visit(node)
        # print("--- debug visit_Call --- ", node.args, node.func)
        if isinstance(node.func, ast.Name):
            if node.func.id in supported_aggr:
                new_node = ast.Attribute(node.args[0], ctx=ast.Load())
                new_node.attr = supported_aggr[node.func.id]
                node.func = new_node
                node.args = []
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        # print("-------------", dir(node))
        # print("visit_IfExp", type(node.test), type(node.body), type(node.orelse))

        if isinstance(node.body, ast.Num):
            raise Exception(
                "Don't support body of IF with number only! Has to be expr with df['column']."
            )

        new_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=node.body, attr="where", ctx=ast.Load()),
                args=[node.test, node.orelse],
                keywords=[],
            )
        )
        # print("-------------")
        # print(astunparse.dump(new_node))
        # print("-------------")

        return new_node

    def visit_Name(self, node):
        self.generic_visit(node)
        print("-------------", node.id)
        if not node.id.startswith("build_in_"):
            new_node = ast.Subscript(
                value=ast.Name(id="df", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=node.id)),
                ctx=ast.Load(),
            )

            node = new_node
        return node


input = "B if ((Aa + cc_01) / 100) == build_in_d else 0"

ast_node = ast.parse(input)
# print(astunparse.dump(ast_node))
transformer = CodeTransformer()
transformer.visit(ast_node)
# print(astunparse.dump(ast_node))
print(astunparse.unparse(ast_node))
