TRANSLATE_PROMPT_BATCH = """
You will translate a list of geometry-related texts (questions, statements, proofs, or answers)
in English or Chinese into natural Vietnamese.

REQUIREMENTS:
- Keep all mathematical symbols exactly as they are (e.g., ∠, ∥, ⟂, =, ≠, ≥, ≤, °, fractions).
- Keep element/point/label names unchanged (e.g., A, B, C, WXZ, TUX, AB, XY).
- Do not rewrite or simplify the symbolic expressions.
- Translate only the English/Chinese words into fluent Vietnamese.
- No explanation. No commentary. No descriptions.

INPUT FORMAT:
You will be given several items in the following numbered format:

1. <text1>
2. <text2>
...
N. <textN>

OUTPUT FORMAT:
Return an array containing N translated strings, in the exact same order.
Example:
["dịch 1", "dịch 2", "..."]

Begin when ready.
"""
