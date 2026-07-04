/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// CUDA Kernel main function

// -----------------------------------------------------------------------------------------

// GPUCompute.h — when USE_SYMMETRY is NOT defined
__device__ void ComputeKangaroos(uint64_t* kangaroos, uint32_t maxFound, uint32_t* out, uint64_t dpMask) {

    // Declare shared memory matrices to hold the jump tables
    // Total size: NB_JUMP * (32 + 32 + 16) bytes
    __shared__ uint64_t s_jPx[NB_JUMP][4];
    __shared__ uint64_t s_jPy[NB_JUMP][4];
    __shared__ uint64_t s_jD[NB_JUMP][2];

    // Cooperatively load jump tables from __constant__ memory to __shared__ memory
    // Every thread in the block processes elements round-robin style
    // BUG FIX 1: Original code assigned s_jPx[i] four times and never assigned
    //            s_jPy[i] or s_jD[i] correctly. Each table entry must be loaded once.
    for (int i = threadIdx.x; i < NB_JUMP; i += blockDim.x) {
        s_jPx[i][0] = jPx[i][0];
        s_jPx[i][1] = jPx[i][1];
        s_jPx[i][2] = jPx[i][2];
        s_jPx[i][3] = jPx[i][3];

        s_jPy[i][0] = jPy[i][0];
        s_jPy[i][1] = jPy[i][1];
        s_jPy[i][2] = jPy[i][2];
        s_jPy[i][3] = jPy[i][3];

        s_jD[i][0] = jD[i][0];
        s_jD[i][1] = jD[i][1];
    }

    // Synchronize to guarantee execution order and ensure data availability
    __syncthreads();

    // Registers allocated per thread execution unit
    // BUG FIX 2: px/py/dist/dx must be 2D arrays of 256-bit (4-limb) and 128-bit (2-limb)
    //            words to match the signatures of LoadKangaroos, StoreKangaroos,
    //            ModSub256, _ModMult, _ModSqr, Load256, and Add128.
    //            Original flat uint64_t[GPU_GRP_SIZE] declarations caused type mismatches
    //            with every function that expected uint64_t(*)[4] or uint64_t(*)[2].
    uint64_t px[GPU_GRP_SIZE][4];
    uint64_t py[GPU_GRP_SIZE][4];
    uint64_t dist[GPU_GRP_SIZE][2];
    uint64_t dx[GPU_GRP_SIZE][4];
    uint64_t dy[4];
    uint64_t rx[4];
    uint64_t ry[4];
    uint64_t _s[4];
    uint64_t _p[4];
    uint32_t jmp;

    // Load states of assigned kangaroos from global into private registers
    LoadKangaroos(kangaroos, px, py, dist);

    // Main iteration loop
    for (int run = 0; run < NB_RUN; run++) {

        __syncthreads();

        // Step 1: Calculate the jump indexes and compute dx = px - s_jPx[jmp]
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            jmp = (uint32_t)px[g][0] & (NB_JUMP - 1);
            ModSub256(dx[g], px[g], s_jPx[jmp]);
        }

        // Step 2: Parallel batch inversion of dx components across the group
        _ModInvGrouped(dx);
        __syncthreads();

        // Step 3: Core elliptic curve step addition using shared memory rows
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            jmp = (uint32_t)px[g][0] & (NB_JUMP - 1);

            // Slope computation calculation steps
            ModSub256(dy, py[g], s_jPy[jmp]);
            _ModMult(_s, dy, dx[g]);
            // BUG FIX 3: _ModSqr takes two pointer arguments (result, input).
            //             With the 2D array fix, _s and _p are now uint64_t[4] as required.
            _ModSqr(_p, _s);

            // rx = _s^2 - px - s_jPx[jmp]
            ModSub256(rx, _p, s_jPx[jmp]);
            ModSub256(rx, rx, px[g]);

            // ry = _s * (px - rx) - py
            ModSub256(ry, px[g], rx);
            _ModMult(ry, ry, _s);
            ModSub256(ry, ry, py[g]);

            // Update registers with new point coordinates
            Load256(px[g], rx);
            Load256(py[g], ry);

            // Accumulate total scalar walk distance
            Add128(dist[g], s_jD[jmp]);

            // Step 4: Check if the coordinate bits match the Distinguished Point (DP) mask
            // BUG FIX 4: px[g] is now uint64_t[4]; the DP mask comparison must use the
            //             lowest limb (px[g][0]) rather than the array itself, which
            //             caused "expression must have pointer-to-object type" errors.
            if ((px[g][3] & dpMask) == 0) {
                uint32_t pos = atomicAdd(out, 1);
                if (pos < maxFound) {
                    uint64_t kIdx = (uint64_t)IDX + (uint64_t)g * (uint64_t)blockDim.x + (uint64_t)blockIdx.x * ((uint64_t)blockDim.x * GPU_GRP_SIZE);
                    OutputDP(px[g], dist[g], &kIdx);
                }
            }
        }
    }

    // Write final results back out to main system global memory
    StoreKangaroos(kangaroos, px, py, dist);
}