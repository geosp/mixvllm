Write a Python function `choose_box(items, boxes)` that picks the smallest shipping box that can hold a set of computer equipment, and returns where each item goes.

Inputs:
- `items`: a list of `(name, length, width, height)` tuples, one per physical item (for example `("laptop", 36, 25, 2)`). Names are unique. Dimensions are positive integers in centimeters. There are at most 6 items.
- `boxes`: a list of `(name, length, width, height)` tuples describing the available box types (positive integers, cm). There are at most 10 box types.

Packing rules:
1. All items must go into a single box.
2. Items are rectangular blocks placed axis-aligned with the box. Each item may be rotated into any of its 6 orientations (any permutation of its three dimensions), but not tilted at an angle.
3. Every item must lie completely inside the box, and no two items may overlap. Items may touch (share a face, edge or corner).
4. Items do not need to be supported from below (no gravity or stacking rules).

Output:
- Choose the box with the smallest volume (`length * width * height`) that can hold all the items under these rules. If several fitting boxes have the same smallest volume, choose the one that appears first in `boxes`.
- Return a tuple `(box_name, placements)`, where `placements` is a list with one entry per item: `(item_name, x, y, z, dx, dy, dz)`. Here `(x, y, z)` is the item's corner closest to the box origin, and `(dx, dy, dz)` is the item's size along the box's length, width and height axes (a permutation of the item's own dimensions). The box spans `0..length` on x, `0..width` on y and `0..height` on z.
- If `items` is empty, return the smallest box (same tie rule) with an empty placement list.
- If no box can hold all the items, or `boxes` is empty, return `None`.

Note that a box having enough total volume does not mean the items fit: the answer must be an actual, valid arrangement. Your function must return the truly smallest fitting box, not just a box that works. Each call must finish within 10 seconds.

Also write unit tests for your implementation.

Constraints: standard library only. Reply with exactly two ```python code blocks:
1. The complete implementation (it will be saved as module `solution`; imports included, no tests, no example usage).
2. Your unit tests: plain `def test_*()` functions using `assert`, importing the code with `from solution import choose_box`. No test framework is needed; each test function will be run and must finish within 10 seconds.

Your tests will be run against your implementation, and any failures will be sent back to you to fix.
