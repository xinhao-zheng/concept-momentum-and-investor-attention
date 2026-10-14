"""Structural source bindings for formatted research kernels."""

from __future__ import annotations
import ast
import copy
import textwrap


def replace_nodes(
    source: str, before: str, after: str, *, required: bool = True
) -> str:
    """Replace an expression or contiguous statements, independent of formatting."""
    tree = ast.parse(source)
    pattern = ast.parse(textwrap.dedent(before).strip()).body
    replacement = ast.parse(textwrap.dedent(after).strip()).body
    matches = 0
    expression = (
        len(pattern) == len(replacement) == 1
        and isinstance(pattern[0], ast.Expr)
        and isinstance(replacement[0], ast.Expr)
    )
    signature = lambda node: ast.dump(node, include_attributes=False)

    class Binder(ast.NodeTransformer):
        def visit(self, node):
            nonlocal matches
            if expression and signature(node) == signature(pattern[0].value):
                matches += 1
                return copy.deepcopy(replacement[0].value)
            return super().visit(node)

        def generic_visit(self, node):
            nonlocal matches
            node = super().generic_visit(node)
            if expression:
                return node
            for field, value in ast.iter_fields(node):
                if not isinstance(value, list) or not all(
                    isinstance(item, ast.stmt) for item in value
                ):
                    continue
                updated, i = [], 0
                while i < len(value):
                    group = value[i : i + len(pattern)]
                    if len(group) == len(pattern) and all(
                        signature(a) == signature(b) for a, b in zip(group, pattern)
                    ):
                        updated.extend(copy.deepcopy(replacement))
                        matches += 1
                        i += len(pattern)
                    else:
                        updated.append(value[i])
                        i += 1
                setattr(node, field, updated)
            return node

    tree = Binder().visit(tree)
    if required and matches == 0:
        raise ValueError("Source binding did not match: " + before)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def bind_assignment(source: str, name: str, expression: str) -> str:
    """Bind one module assignment while preserving all other AST nodes."""
    tree = ast.parse(source)
    matches = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            node.value = ast.parse(expression, mode="eval").body
            matches += 1
    if matches != 1:
        raise ValueError(f"Expected one module assignment for {name}; found {matches}")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def bind_threads(source: str, threads: int) -> str:
    """Bind numerical thread settings and their metadata; preserve model settings."""
    tree = ast.parse(source)

    class Threads(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "threadpool_limits":
                for keyword in node.keywords:
                    if (
                        keyword.arg == "limits"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == 4
                    ):
                        keyword.value = ast.Constant(threads)
            for keyword in node.keywords:
                if (
                    keyword.arg == "max_workers"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == 4
                ):
                    keyword.value = ast.Constant(threads)
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Constant) and node.value.value == "4":
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id == "os"
                        and target.value.attr == "environ"
                    ):
                        node.value = ast.Constant(str(threads))
            return node

        def visit_Dict(self, node):
            node = self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "threads"
                    and isinstance(node.values[i], ast.Constant)
                    and node.values[i].value == 4
                ):
                    node.values[i] = ast.Constant(threads)
            return node

    return ast.unparse(ast.fix_missing_locations(Threads().visit(tree))) + "\n"


def bind_paths(source: str, bindings: dict) -> str:
    """Bind module-level Path values; preserve identically named data objects."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "Path":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in bindings:
                node.value = ast.parse(
                    "Path(" + repr(str(bindings[target.id])) + ")", mode="eval"
                ).body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
