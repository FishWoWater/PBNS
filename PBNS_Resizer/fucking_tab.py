#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp 

for _root, _dirs, _files in os.walk("."):
    for _file in _files:
        if not _file.endswith('.py'):   continue
        path = osp.join(_root, _file)
        fin = open(path, "r")
        content = fin.read()
        content = content.replace('\t', '    ')
        print(content)
        fin.close()
        fout = open(path, "w")
        fout.write(content)
        fout.close()
