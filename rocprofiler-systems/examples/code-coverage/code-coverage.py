#!@PYTHON_EXECUTABLE@

import rocprofsys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Input code coverage",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output code coverage",
        default=None,
        required=True,
    )

    args = parser.parse_args()

    data = None
    for itr in args.input:
        _summary, _details = rocprofsys.coverage.load(itr)
        if data is None:
            data = _details
        else:
            data = rocprofsys.coverage.concat(data, _details)

    summary = rocprofsys.coverage.get_summary(data)
    top = rocprofsys.coverage.get_top(data)
    bottom = rocprofsys.coverage.get_bottom(data)

    print("Top code coverage:")
    for itr in top:
        print(
            f"    {itr.count} | {itr.function} | {itr.module}:{itr.line} | {itr.source}"
        )

    print("Bottom code coverage:")
    for itr in bottom:
        print(
            f"    {itr.count} | {itr.function} | {itr.module}:{itr.line} | {itr.source}"
        )

    print("\nSaving code coverage")
    rocprofsys.coverage.save(summary, data, args.output)
