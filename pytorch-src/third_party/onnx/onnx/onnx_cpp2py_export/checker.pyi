from typing import Dict, Union


class CheckerContext:
    ir_version: int = ...
    opset_imports: Dict[str, int] = ...


class ValidationError(Exception):
    ...


def check_value_info(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_tensor(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_sparse_tensor(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_attribute(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_node(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_graph(bytes: bytes, checker_context: CheckerContext) -> None: ...
def check_model(bytes: bytes) -> None: ...
def check_model_path(path: str) -> None: ...
