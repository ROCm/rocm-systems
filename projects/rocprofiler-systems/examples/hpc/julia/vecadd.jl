# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Vector addition example using AMDGPU.jl
using AMDGPU

# GPU kernel: element-wise vector addition
function vadd!(c, a, b)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    if i ≤ length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

function main()
    println("Starting Julia AMDGPU vector addition...")

    # Vector size
    n = 1024

    # Create arrays on GPU
    a = ROCArray(rand(Float32, n))
    b = ROCArray(rand(Float32, n))
    c = ROCArray(zeros(Float32, n))

    # Launch kernel
    groupsize = 256
    gridsize = cld(n, groupsize)

    @roc groupsize=groupsize gridsize=gridsize vadd!(c, a, b)
    AMDGPU.synchronize()

    println("PASSED!")
end

main()
