"""
Compiler Grader for SweetPea DSL Code Generation.

Executes candidate model-generated Python code in an isolated environment,
verifying:
1. Syntax validity (AST parse)
2. SweetPea module imports and construct definitions
3. Runtime execution and trial synthesis (CryptoMiniSat / SAT solver)
4. Constraint satisfaction and experiment shape
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class GradeResult:
    passed: bool
    score: float  # 0.0 to 1.0 stepped reward
    syntax_valid: bool
    ast_parsed: bool
    imports_valid: bool
    block_created: bool
    synthesized: bool
    num_trials: int
    num_factors: int
    error_type: Optional[str]
    error_message: Optional[str]
    execution_time_sec: float
    raw_output: str


def extract_python_code(text: str) -> str:
    """Extracts python code from markdown code fences or returns clean text."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def check_ast(code: str) -> Tuple[bool, bool, Optional[str]]:
    """Checks if code is valid Python and imports sweetpea."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, False, f"SyntaxError: {e.msg} at line {e.lineno}"

    imports_sp = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "sweetpea" in alias.name:
                    imports_sp = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and "sweetpea" in node.module:
                imports_sp = True

    return True, imports_sp, None


# Wrapper template to execute user code and introspect the generated block/experiments
HARNESS_TEMPLATE = """
import sys
import json
import traceback
import time

try:
    import sweetpea as sp
except ImportError as e:
    print(json.dumps({
        "status": "error",
        "error_type": "ImportError",
        "error_message": str(e),
        "score": 0.1
    }))
    sys.exit(0)

start_time = time.perf_counter()

# User code follows:
try:
{USER_CODE}
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({
        "status": "error",
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": tb,
        "score": 0.3
    }))
    sys.exit(0)

exec_time = time.perf_counter() - start_time

# Look for block or experiments in local/global scope
locs = locals()
globs = globals()

block_obj = None
experiments_obj = None

for scope in [locs, globs]:
    for k, v in list(scope.items()):
        if isinstance(v, (sp.CrossBlock, sp.MultiCrossBlock, sp.Repeat)):
            block_obj = v
        elif k == "experiments" or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)):
            experiments_obj = v
        elif k == "block" and block_obj is None:
            block_obj = v

# If experiments not yet synthesized, attempt synthesis if block exists
if experiments_obj is None and block_obj is not None:
    try:
        experiments_obj = sp.synthesize_trials(block_obj, 1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error_type": f"SynthesisError_{type(e).__name__}",
            "error_message": str(e),
            "block_created": True,
            "synthesized": False,
            "score": 0.6,
            "execution_time_sec": exec_time
        }))
        sys.exit(0)

if experiments_obj is not None and len(experiments_obj) > 0:
    first_exp = experiments_obj[0]
    num_factors = len(first_exp.keys()) if isinstance(first_exp, dict) else 0
    num_trials = len(list(first_exp.values())[0]) if isinstance(first_exp, dict) and num_factors > 0 else 0
    print(json.dumps({
        "status": "success",
        "block_created": True,
        "synthesized": True,
        "num_factors": num_factors,
        "num_trials": num_trials,
        "score": 1.0,
        "execution_time_sec": exec_time
    }))
else:
    print(json.dumps({
        "status": "incomplete",
        "error_type": "NoExperimentsProduced",
        "error_message": "Code executed without error but did not construct a CrossBlock or synthesize trials.",
        "block_created": block_obj is not None,
        "synthesized": False,
        "score": 0.4,
        "execution_time_sec": exec_time
    }))
"""


def grade_sweetpea_code(
    code_or_completion: str,
    timeout_sec: float = 10.0,
    python_executable: str = sys.executable,
) -> GradeResult:
    """Grades candidate SweetPea Python code.

    Returns structured GradeResult with execution verification and stepped
    reward.
    """
    code = extract_python_code(code_or_completion)

    # 1. AST Check
    syntax_ok, imports_ok, syntax_err = check_ast(code)
    if not syntax_ok:
        return GradeResult(
            passed=False,
            score=0.0,
            syntax_valid=False,
            ast_parsed=False,
            imports_valid=False,
            block_created=False,
            synthesized=False,
            num_trials=0,
            num_factors=0,
            error_type="SyntaxError",
            error_message=syntax_err,
            execution_time_sec=0.0,
            raw_output="",
        )

    # 2. Indent user code for the wrapper
    indented_code = "\n".join("    " + line for line in code.splitlines())
    script_content = HARNESS_TEMPLATE.replace("{USER_CODE}", indented_code)

    # 3. Write to temporary file and execute in subprocess
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script_content)
        temp_path = f.name

    try:
        res = subprocess.run(
            [python_executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()

        # Parse JSON output from last line of stdout
        parsed_json = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed_json = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed_json:
            status = parsed_json.get("status")
            score = float(parsed_json.get("score", 0.0))
            passed = status == "success" and score >= 0.99
            return GradeResult(
                passed=passed,
                score=score,
                syntax_valid=True,
                ast_parsed=True,
                imports_valid=imports_ok,
                block_created=parsed_json.get("block_created", False),
                synthesized=parsed_json.get("synthesized", False),
                num_trials=parsed_json.get("num_trials", 0),
                num_factors=parsed_json.get("num_factors", 0),
                error_type=parsed_json.get("error_type"),
                error_message=parsed_json.get("error_message"),
                execution_time_sec=parsed_json.get("execution_time_sec", 0.0),
                raw_output=stdout,
            )
        else:
            return GradeResult(
                passed=False,
                score=0.2,
                syntax_valid=True,
                ast_parsed=True,
                imports_valid=imports_ok,
                block_created=False,
                synthesized=False,
                num_trials=0,
                num_factors=0,
                error_type="SubprocessCrash",
                error_message=stderr or stdout,
                execution_time_sec=0.0,
                raw_output=stdout + "\n" + stderr,
            )

    except subprocess.TimeoutExpired:
        return GradeResult(
            passed=False,
            score=0.1,
            syntax_valid=True,
            ast_parsed=True,
            imports_valid=imports_ok,
            block_created=False,
            synthesized=False,
            num_trials=0,
            num_factors=0,
            error_type="TimeoutExpired",
            error_message=f"Execution exceeded timeout of {timeout_sec}s",
            execution_time_sec=timeout_sec,
            raw_output="",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    sample_code = """
import sweetpea as sp

color = sp.Factor("color", ["red", "blue"])
motion = sp.Factor("motion", ["up", "down"])
block = sp.CrossBlock([color, motion], [color, motion], [])
experiments = sp.synthesize_trials(block, 1)
"""
    result = grade_sweetpea_code(sample_code)
    print("Self-test result:", asdict(result))
