Write a Python function `evaluate(expr: str) -> Fraction` that evaluates an arithmetic expression exactly and returns a `fractions.Fraction`.

Specification:

1. Tokens: non-negative number literals, the operators `+ - * / ^`, parentheses `( )`, and whitespace (spaces only, ignored between tokens). Anything else is an error.
2. Number literals are one or more digits, optionally followed by `.` and one or more digits (e.g. `7`, `3.25`, `0.1`). Forms like `3.`, `.5`, `1..2`, `1e3` are errors. Literals must be converted exactly (so `0.1 + 0.2` is exactly `3/10`).
3. Precedence from lowest to highest:
   - binary `+` and `-` (left-associative)
   - binary `*` and `/` (left-associative)
   - unary `+` and `-` (prefix, may repeat, e.g. `--3` is 3)
   - `^` exponentiation (right-associative: `2^3^2` is 512)
   Because unary minus binds looser than `^`, `-2^2` is -4 and `(-2)^2` is 4. The right operand of `^` is itself a unary expression, so `2^-1` is 1/2 and `2^-2^2` is 2^(-(2^2)) = 1/16.
4. The exponent must evaluate to an integer value (`4^1.0` is 4, but `2^0.5` and `2^(1/2)` are errors). Negative integer exponents are allowed. `0^0` is 1.
5. There is no implicit multiplication: `2(3)` and `1 2` are errors. `**` is not an operator.
6. Errors: raise `ValueError` for every invalid input, including empty or whitespace-only input, unbalanced parentheses, unexpected or missing tokens, invalid literals, non-integer exponents, and any division by zero (including `0^-1`). Do not let `ZeroDivisionError` or any other exception type escape.

Constraints: standard library only, no `eval`/`exec`/`ast`/`compile`. Reply with the complete implementation in a single ```python code block (imports included, no tests, no example usage).
