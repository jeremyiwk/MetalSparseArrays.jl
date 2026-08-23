"""
    DEFAULT_INDEX_TYPE

The integer type used for row, column, and pointer arrays of device-resident sparse
matrices, `Int32`. This matches the index type of `CUDA.CUSPARSE` and avoids the
memory and bandwidth cost of 64-bit indices, at the cost of restricting a stored
dimension or nonzero count to `typemax(Int32)`. `SparseArrays.SparseMatrixCSC`
defaults to `Int` on the CPU, so conversion between host and device changes `Ti`.
"""
const DEFAULT_INDEX_TYPE = Int32

"""
    indextype(A)

The integer type `Ti` used for the index arrays of the sparse array `A`.
"""
indextype(::AbstractSparseArray{<:Any, Ti}) where {Ti} = Ti
indextype(::Type{<:AbstractSparseArray{<:Any, Ti}}) where {Ti} = Ti

"""
    realtype(T)

The real floating point type used for magnitudes of elements of type `T`, i.e.
`real(float(T))`. Used for norms, tolerances, and convergence criteria.
"""
realtype(::Type{T}) where {T <: Number} = real(float(T))
realtype(x) = realtype(typeof(x))

"""
    unit_roundoff(T)

The unit roundoff `u = eps(realtype(T)) / 2` of the floating point type associated
with `T`.
"""
unit_roundoff(::Type{T}) where {T <: Number} = eps(realtype(T)) / 2
unit_roundoff(x) = unit_roundoff(typeof(x))
