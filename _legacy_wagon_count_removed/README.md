# REMOVED legacy wagon-counting engine (review copy only)

These are the **replaced** wagon-counting modules from the previous
`wagon_count/` package, kept here **only so the counting-engine swap can be
reviewed**. They are dead code.

* This directory is **not a Python package** (no `__init__.py`).
* Nothing in `wagon_eye_v4/` imports it.
* Stage 1 resolves `wagon_count/run_global_count.py` only
  (`reconstruction/runner.py::_find_wagon_count_dir`), and the subprocess runs
  with `cwd=wagon_count/`, so these files are never on `sys.path`.
* `tests/test_counting_engine_swap.py` asserts all of the above.

Delete this directory once the review is complete.
