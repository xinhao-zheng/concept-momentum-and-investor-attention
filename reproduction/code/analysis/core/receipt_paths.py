"""Serialize and resolve source and output paths in calculation receipts."""

from pathlib import Path


def receipt_path(path, source_root, output_root):
    """Prefer a logical root; retain an absolute path for unrelated dependencies."""
    path = Path(path).resolve()
    # The default output is inside source_root, so check output first.
    for label, root in [("output", output_root), ("source", source_root)]:
        try:
            relative = path.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return {"path_root": label, "path": relative.as_posix()}
    return {"path_root": "absolute", "path": str(path)}


def resolve_receipt_path(reference, source_root, output_root):
    """Resolve new references, plus historical PACKAGE-relative references."""
    kind = reference.get("path_root", "source")
    path = Path(reference["path"])
    if kind == "absolute":
        if not path.is_absolute():
            raise ValueError("Absolute receipt reference must contain an absolute path")
        return path.resolve()
    roots = {"source": source_root, "output": output_root}
    if kind not in roots:
        raise ValueError("Unknown receipt path root: " + str(kind))
    root = Path(roots[kind]).resolve()
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("Logical receipt path must remain within its declared root")
    return resolved


def receipt_transform(source):
    """Adapt the two receipt expressions only; fail if the source contract drifts."""
    old = "path=str(path.relative_to(PACKAGE))"
    if source.count(old) != 1:
        raise ValueError("Expected exactly one PACKAGE-relative receipt serializer")
    source = source.replace(old, "**_receipt_path(path)")
    candidates = ["PACKAGE/item['path']", "PACKAGE/inp['path']"]
    if sum(source.count(expr) for expr in candidates) != 1:
        raise ValueError("Expected exactly one receipt hash lookup")
    for expression, replacement in zip(
        candidates, ["_resolve_receipt_path(item)", "_resolve_receipt_path(inp)"]
    ):
        source = source.replace(expression, replacement)
    return source


def bind_receipt_paths(module, source_root, output_root):
    module._receipt_path = lambda path: receipt_path(path, source_root, output_root)
    module._resolve_receipt_path = lambda reference: resolve_receipt_path(
        reference, source_root, output_root
    )
    return module
