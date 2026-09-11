"""查 pycolmap 的 options 对象有哪些可调字段。"""
import os

import pycolmap
from pycolmap import FeatureExtractionOptions, FeatureMatchingOptions, IncrementalPipelineOptions

print("pycolmap", pycolmap.__version__, "| 逻辑核数", os.cpu_count())
print()

def dump(name, obj):
    print(f"===== {name} =====")
    try:
        d = obj.summary() if hasattr(obj, "summary") else None
    except Exception:
        d = None
    if d:
        print(d)
        return
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(obj, attr)
        except Exception:
            continue
        if callable(v):
            continue
        print(f"    {attr} = {v}")
    print()

dump("FeatureExtractionOptions", FeatureExtractionOptions())
dump("FeatureMatchingOptions", FeatureMatchingOptions())
dump("IncrementalPipelineOptions", IncrementalPipelineOptions())

print("Device 可选值:", [x for x in dir(pycolmap.Device) if not x.startswith("_")])
