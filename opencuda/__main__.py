"""
OpenCUDA compiler — CUDA-subset C to SM_120 cubin.

Usage:
    python -m opencuda kernel.cu [--out kernel.cubin] [--emit-ptx] [-v]
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        prog="opencuda",
        description="Open-source CUDA compiler — C to SM_120 cubin",
    )
    ap.add_argument("source", nargs="?", help="Input .cu file")
    ap.add_argument("--out", default=None, help="Output .cubin")
    ap.add_argument("--emit-ptx", action="store_true",
                    help="Emit PTX text instead of cubin")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.source is None:
        ap.print_help()
        sys.exit(0)

    source_path = Path(args.source)
    source = source_path.read_text(encoding='utf-8')

    # Parse C → IR
    from opencuda.frontend.parser import parse
    print(f"[opencuda] Parsing {args.source}...")
    module = parse(source)
    print(f"[opencuda] {len(module.kernels)} kernel(s) found")

    # IR → PTX
    from opencuda.codegen.emit import ir_to_ptx
    ptx_map = ir_to_ptx(module)

    if args.emit_ptx:
        # Concatenate all kernels into one PTX module
        all_ptx = []
        all_ptx.append('.version 9.0')
        all_ptx.append('.target sm_120')
        all_ptx.append('.address_size 64')
        all_ptx.append('')
        for kernel_name, ptx_text in ptx_map.items():
            # Strip the header (already added above) and append kernel body
            lines = ptx_text.split('\n')
            # Skip .version, .target, .address_size lines
            body_start = 0
            for j, line in enumerate(lines):
                if line.startswith('.visible') or line.startswith('{'):
                    body_start = j
                    break
            all_ptx.extend(lines[body_start:])
            all_ptx.append('')
        out = args.out or str(source_path.with_suffix('.ptx'))
        Path(out).write_text('\n'.join(all_ptx), encoding='utf-8')
        print(f"[opencuda] Wrote PTX: {out} ({len(ptx_map)} kernels)")
        if args.verbose:
            print('\n'.join(all_ptx))
    else:
        for kernel_name, ptx_text in ptx_map.items():
            # PTX → cubin via OpenPTXas
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'openptxas'))
            from sass.pipeline import compile_ptx_source
            print(f"[opencuda] Compiling kernel '{kernel_name}' via OpenPTXas...")
            cubins = compile_ptx_source(ptx_text, verbose=args.verbose)
            for kname, cubin_bytes in cubins.items():
                out = args.out or str(source_path.with_suffix('.cubin'))
                Path(out).write_bytes(cubin_bytes)
                print(f"[opencuda] Wrote cubin: {out} ({len(cubin_bytes)} bytes)")

    print("[opencuda] Done.")


if __name__ == "__main__":
    main()
