---
name: Bug report
about: A crash, a wrong result, or a mismatch with the CPU reference
---

**Describe the bug**

What happened, and what you expected instead. For numerical mismatches, state
the element type, index type, format, problem size, and the observed versus
expected difference from the `SparseArrays` result.

**Minimal reproducer**

```julia
# The smallest script that shows the problem.
```

**Environment**

Bugs here are frequently hardware- and driver-dependent, so this section
matters. Paste the output of:

```julia
using Pkg; Pkg.status("MetalSparseArrays")
versioninfo()
using Metal; Metal.versioninfo()
```
