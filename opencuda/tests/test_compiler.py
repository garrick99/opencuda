"""
Automated regression tests for OpenCUDA compiler.

Tests that all .cu files in tests/ compile to valid PTX accepted by ptxas.
"""

import os
import subprocess
import pytest
from pathlib import Path

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'
CU_FILES = sorted(TESTS_DIR.glob('*.cu'))

# Filter out test harness files (they're C++ with #include, not kernels)
KERNEL_FILES = [f for f in CU_FILES if not f.name.startswith('gpu_')]


@pytest.fixture(params=KERNEL_FILES, ids=[f.stem for f in KERNEL_FILES])
def cu_file(request):
    return request.param


def test_parse_and_emit(cu_file):
    """Each .cu file should parse and emit valid PTX."""
    source = cu_file.read_text(encoding='utf-8')
    source = preprocess(source)
    module = parse(source)
    assert len(module.kernels) >= 1, f"No kernels found in {cu_file.name}"
    module = optimize(module)
    ptx_map = ir_to_ptx(module)
    assert len(ptx_map) >= 1, f"No PTX emitted for {cu_file.name}"

    for kernel_name, ptx_text in ptx_map.items():
        assert '.version' in ptx_text
        assert '.entry' in ptx_text
        assert 'ret;' in ptx_text


def test_ptxas_validates(cu_file, tmp_path):
    """Each .cu file should produce PTX that ptxas accepts."""
    source = cu_file.read_text(encoding='utf-8')
    source = preprocess(source)
    module = parse(source)
    module = optimize(module)
    ptx_map = ir_to_ptx(module)

    # Concatenate all kernels
    all_ptx = ['.version 9.0', '.target sm_120', '.address_size 64', '']
    for ptx_text in ptx_map.values():
        lines = ptx_text.split('\n')
        for j, line in enumerate(lines):
            if line.startswith('.visible') or line.startswith('{'):
                all_ptx.extend(lines[j:])
                break
        all_ptx.append('')

    ptx_file = tmp_path / f"{cu_file.stem}.ptx"
    ptx_file.write_text('\n'.join(all_ptx), encoding='utf-8')

    out_file = tmp_path / f"{cu_file.stem}.cubin"
    result = subprocess.run(
        ['ptxas', '-arch', 'sm_120', str(ptx_file), '-o', str(out_file)],
        capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, f"ptxas rejected {cu_file.name}:\n{result.stdout}\n{result.stderr}"


def test_constant_folding():
    """Constant folding should reduce 2*16 to 32."""
    source = '''
__global__ void fold_test(int *out) {
    int x = 2 * 16;
    int y = x + 0;
    out[0] = y;
}
'''
    module = parse(preprocess(source))
    from opencuda.ir.optimize import constant_fold
    n = constant_fold(module.kernels[0])
    assert n >= 2, f"Expected at least 2 folds, got {n}"


def test_preprocessor():
    """#define should substitute values."""
    source = '#define FOO 42\n__global__ void test(int *out) { out[0] = FOO; }\n'
    processed = preprocess(source)
    assert 'FOO' not in processed.split('\n')[-1]  # FOO should be replaced
    assert '42' in processed
