# Changelog

All notable changes to this package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Restore Int64 pointer scans and exercise both supported index types.
- Fix in-place broadcast from device views and wrappers and host CSC conversion
  with oversized storage tails.
- Make copy, similar and unary broadcast use asynchronous pattern copies without
  redundant validation. Expand CSC findnz indices on device.
- Densify directly from each sparse format without coordinate conversion or an
  intermediate linear-index array.
- Add CPU/device array-interface benchmarks with latency distributions.
- Validate host storage before upload, use one scan launch for small inputs, and
  avoid count readback for structurally empty merges.
- Run zero-preserving self broadcast assignment on device while retaining
  SparseArrays' numerical-zero compaction semantics.

Initial development. Package scaffolding, the device-guarded test harness
validating against the `SparseArrays` CPU reference, CI (tests, Aqua QA,
ExplicitImports, Runic formatting, docs, benchmarks), and shared utilities
(`DEFAULT_INDEX_TYPE`, `indextype`, `realtype`, `unit_roundoff`).
