"""Skill 1: calculator - safe arithmetic without eval()."""

import ast
import operator

from . import skill

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPONENT = 1000
MAX_LENGTH = 200


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_EXPONENT})")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _format(value: float) -> str:
    if isinstance(value, float):
        value = round(value, 10)  # hide float noise like 397.80000000000001
        if value.is_integer():
            return str(int(value))
    return str(value)


@skill(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression and return the exact result. "
        "Supports + - * / // % ** and parentheses on numbers. "
        "Use this for any math instead of calculating in your head. "
        "Example: '17/100*2340'."
    ),
    properties={
        "expression": {"type": "string", "description": "The arithmetic expression to evaluate."}
    },
)
def calculator(expression: str) -> str:
    if len(expression) > MAX_LENGTH:
        raise ValueError(f"Expression too long (max {MAX_LENGTH} characters)")
    tree = ast.parse(expression.replace(",", ""), mode="eval")
    return _format(_eval(tree.body))
