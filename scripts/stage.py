#!/usr/bin/env python3

import dataclasses as dc
import os
import re
import sys


def sanitize_fname(fname: str):
    fixed = fname.replace(" ", "-")
    return fixed


@dc.dataclass
class FileRef:
    orig_name: str
    sanitized_name: str
    abspath: str
    absdir: str
    ext: str

    @staticmethod
    def parse(file_path: str) -> "FileRef":
        rel_name, ext = os.path.splitext(file_path)
        ext = ext[1:]
        orig_name = os.path.basename(rel_name)
        absdir = os.path.abspath(
            os.path.dirname(rel_name),
        )
        abspath = f"{os.path.join(absdir, orig_name)}.{ext}" if ext else \
            os.path.join(absdir, orig_name)
        return FileRef(
            sanitized_name=sanitize_fname(os.path.basename(rel_name)),
            orig_name=orig_name,
            abspath=abspath,
            absdir=absdir,
            ext=ext,
        )


def run(cmd: list[str]):
    print(" ".join(cmd))
    os.system(" ".join(cmd))
    return


def main():
    if len(sys.argv) < 2:
        print("need target")
        sys.exit(1)

    target = FileRef.parse(sys.argv[1])
    assert target.ext == "md", "should be a markdown file"


    posts_fref = FileRef.parse(
        os.path.join(target.absdir, "..", "_posts", f"{target.orig_name}.{target.ext}")
    )

    files_fref = FileRef.parse(
        os.path.join(target.absdir, f"{target.orig_name}_files")
    )

    run([
        "cp", target.abspath, posts_fref.abspath,
    ])
    run([
        "cp", "-r", files_fref.abspath, os.path.join(files_fref.absdir, "..", "assets"),
    ])

    print(f"updating links in '{posts_fref.abspath}'")
    lines = []
    with open(posts_fref.abspath, "r") as f:
        lines = f.read().split("\n")

    adjusted_lines = []
    for line in lines:
        if line.startswith(f"![png]({files_fref.orig_name}"):
            _, rest = line.split("![png](", 1)
            adjusted_lines.append("".join([
                "![png](",
                "/assets/",
                rest,
            ]))
        else:
            adjusted_lines.append(line)

    with open(posts_fref.abspath, "w") as f:
        f.write("\n".join(adjusted_lines))
    return


if __name__ == '__main__':
    main()
