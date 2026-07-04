/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
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

#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>

#include <thread>
#include <chrono>

#include <cstdio>      // Required for printf
#include <cstdint>     // Required for uint64_t
#include <inttypes.h>  // Required for the PRIX64 macro


#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}


void Kangaroo::PrintDPDebug(Int* x, Int* distance, uint32_t kType, int threadId, const char* mode) {
    printf("[DEBUG DP] Thread %d (%s): Found DP\n", threadId, mode);
    Point PR = secp->ComputePublicKey(distance);

    printf("  Pub:  %s \n", PR.x.GetBase16().c_str());
    printf("  X  : %064s\n", x->GetBase16().c_str());
    printf("  D  : %s\n", distance->GetBase16().c_str());
    printf("  Type: %s\n", (kType == TAME) ? "TAME" : "WILD");
    printf("\n");
}

// ----------------------------------------------------------------------------

Kangaroo::Kangaroo(Secp256K1* secp, int32_t initDPSize, bool useGpu, std::string& workFile, std::string& iWorkFile,
    uint32_t savePeriod, bool saveKangaroo, bool saveKangarooByServer,
    double maxStep, int wtimeout, int port, int ntimeout, std::string serverIp,
    std::string outputFile, bool splitWorkfile) {

    this->secp = secp;
    this->initDPSize = initDPSize;
    this->useGpu = useGpu;
    this->offsetCount = 0;
    this->offsetTime = 0.0;
    this->workFile = workFile;
    this->saveWorkPeriod = savePeriod;
    this->inputFile = iWorkFile;
    this->nbLoadedWalk = 0;
    this->clientMode = serverIp.length() > 0;
    this->saveKangarooByServer = this->clientMode && saveKangarooByServer;
    this->saveKangaroo = saveKangaroo || this->saveKangarooByServer;
    this->fRead = NULL;
    this->maxStep = maxStep;
    this->wtimeout = wtimeout;
    this->port = port;
    this->ntimeout = ntimeout;
    this->serverIp = serverIp;
    this->outputFile = outputFile;
    this->hostInfo = NULL;
    this->endOfSearch = false;
    this->saveRequest = false;
    this->connectedClient = 0;
    this->totalRW = 0;
    this->collisionInSameHerd = 0;
    this->keyIdx = 0;
    this->splitWorkfile = splitWorkfile;
    this->pid = Timer::getPID();

    _generateTameMode = false;
    _generateTameFile = "";
    _generateTameSavePeriod = 0;

    // Wilds-only mode initialisation
    wildsOnlyMode = false;
    wildsOffsetBits = 0;
    wildsOutputFile = "wilds.txt";
    this->lastStatusTime = 0.0;
    this->startTime = 0.0;

    // Sequential starting-point tracker for --wilds mode
    wildsNextStart.SetInt32(0);
    wildsStep.SetInt32(0);  // set properly in Run() once wildsOnlyMode is confirmed
    lastKernelMs = 0.0;

    // Subrange tracker for --wilds mode when wildsStepBits == 0
    wildsStepBits = 0;
    wildsSubRangeMode = false;
    wildsNbSubRanges = 0;
    wildsSubRangeIdx = 0;
    wildsSubRangeBits = 0;

    CPU_GRP_SIZE = 1024;

    // Init mutexes
#ifdef WIN64
    ghMutex = CreateMutex(NULL, FALSE, NULL);
    saveMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_mutex_init(&ghMutex, NULL);
    pthread_mutex_init(&saveMutex, NULL);
    signal(SIGPIPE, SIG_IGN);
#endif
}

Kangaroo::~Kangaroo() {
    // Destroy mutexes
#ifdef WIN64
    if (ghMutex) CloseHandle(ghMutex);
    if (saveMutex) CloseHandle(saveMutex);
#else
    pthread_mutex_destroy(&ghMutex);
    pthread_mutex_destroy(&saveMutex);
#endif
    // Close work file if open
    if (fRead) {
        fclose(fRead);
        fRead = NULL;
    }
}

// ----------------------------------------------------------------------------

void Kangaroo::SetGenerateTameMode(const std::string& fileName, uint32_t savePeriod) {
    _generateTameMode = true;
    _generateTameFile = fileName;
    _generateTameSavePeriod = savePeriod;
}

// ---------------

bool Kangaroo::ParseConfigFile(std::string& fileName) {

    // In client mode, config come from the server
    if (clientMode)
        return true;

    // Check file
    FILE* fp = fopen(fileName.c_str(), "rb");
    if (fp == NULL) {
        ::printf("Error: Cannot open %s %s\n", fileName.c_str(), strerror(errno));
        return false;
    }
    fclose(fp);

    // Get lines
    vector<string> lines;
    int nbLine = 0;
    string line;
    ifstream inFile(fileName);
    while (getline(inFile, line)) {

        // Remove ending \r\n
        int l = (int)line.length() - 1;
        while (l >= 0 && isspace(line.at(l))) {
            line.pop_back();
            l--;
        }

        if (line.length() > 0) {
            lines.push_back(line);
            nbLine++;
        }

    }

    if (lines.size() < 3) {
        ::printf("Error: %s not enough arguments\n", fileName.c_str());
        return false;
    }

    rangeStart.SetBase16((char*)lines[0].c_str());
    rangeEnd.SetBase16((char*)lines[1].c_str());
    for (int i = 2;i < (int)lines.size();i++) {

        Point p;
        bool isCompressed;
        if (!secp->ParsePublicKeyHex(lines[i], p, isCompressed)) {
            ::printf("%s, error line %d: %s\n", fileName.c_str(), i, lines[i].c_str());
            return false;
        }
        keysToSearch.push_back(p);

    }

    ::printf("Start:%s\n", rangeStart.GetBase16().c_str());
    ::printf("Stop :%s\n", rangeEnd.GetBase16().c_str());
    ::printf("Keys :%d\n", (int)keysToSearch.size());

    return true;

}

// ----------------------------------------------------------------------------

bool Kangaroo::IsDP(uint64_t x) {

    return (x & dMask) == 0;

}

void Kangaroo::SetDP(int size) {

    // Mask for distinguised point
    dpSize = size;
    if (dpSize == 0) {
        dMask = 0;
    }
    else {
        if (dpSize > 64) dpSize = 64;
        dMask = (1ULL << (64 - dpSize)) - 1;
        printf("DP mask: 0x%" PRIX64 "\n", dMask);
        dMask = ~dMask;
    }

#ifdef WIN64
    ::printf("DP size: %d [0x%016I64X]\n", dpSize, dMask);
#else
    ::printf("DP size: %d [0x%" PRIx64 "]\n", dpSize, dMask);
#endif

}

// ----------------------------------------------------------------------------

bool Kangaroo::Output(Int* pk, char sInfo, int sType) {


    FILE* f = stdout;
    bool needToClose = false;

    if (outputFile.length() > 0) {
        f = fopen(outputFile.c_str(), "a");
        if (f == NULL) {
            printf("Cannot open %s for writing\n", outputFile.c_str());
            f = stdout;
        }
        else {
            needToClose = true;
        }
    }

    if (!needToClose)
        ::printf("\n");

    Point PR = secp->ComputePublicKey(pk);

    ::fprintf(f, "Key#%2d [%d%c]Pub:  0x%s \n", keyIdx, sType, sInfo, secp->GetPublicKeyHex(true, keysToSearch[keyIdx]).c_str());
    if (PR.equals(keysToSearch[keyIdx])) {
        ::fprintf(f, "       Priv: 0x%s \n", pk->GetBase16().c_str());
    }
    else {
        ::fprintf(f, "       Failed !\n");
        if (needToClose)
            fclose(f);
        return false;
    }


    if (needToClose)
        fclose(f);

    return true;

}

// ----------------------------------------------------------------------------

bool  Kangaroo::CheckKey(Int d1, Int d2, uint8_t type) {

    // Resolve equivalence collision

    if (type & 0x1)
        d1.ModNegK1order();
    if (type & 0x2)
        d2.ModNegK1order();

    Int pk(&d1);
    pk.ModAddK1order(&d2);

    Point P = secp->ComputePublicKey(&pk);

    if (P.equals(keyToSearch)) {
        // Key solved    
#ifdef USE_SYMMETRY
        pk.ModAddK1order(&rangeWidthDiv2);
#endif
        pk.ModAddK1order(&rangeStart);
        return Output(&pk, 'N', type);
    }

    if (P.equals(keyToSearchNeg)) {
        // Key solved
        pk.ModNegK1order();
#ifdef USE_SYMMETRY
        pk.ModAddK1order(&rangeWidthDiv2);
#endif
        pk.ModAddK1order(&rangeStart);
        return Output(&pk, 'S', type);
    }

    return false;

}

bool Kangaroo::CollisionCheck(Int* d1, uint32_t type1, Int* d2, uint32_t type2) {


    if (type1 == type2) {

        // Collision inside the same herd
        return false;

    }
    else {

        Int Td;
        Int Wd;

        if (type1 == TAME) {
            Td.Set(d1);
            Wd.Set(d2);
        }
        else {
            Td.Set(d2);
            Wd.Set(d1);
        }

        endOfSearch = CheckKey(Td, Wd, 0) || CheckKey(Td, Wd, 1) || CheckKey(Td, Wd, 2) || CheckKey(Td, Wd, 3);

        if (!endOfSearch) {

            // Should not happen, reset the kangaroo
            ::printf("\n Unexpected wrong collision, reset kangaroo !\n");
            if ((int64_t)(Td.bits64[3]) < 0) {
                Td.ModNegK1order();
                ::printf("Found: Td-%s\n", Td.GetBase16().c_str());
            }
            else {
                ::printf("Found: Td %s\n", Td.GetBase16().c_str());
            }
            if ((int64_t)(Wd.bits64[3]) < 0) {
                Wd.ModNegK1order();
                ::printf("Found: Wd-%s\n", Wd.GetBase16().c_str());
            }
            else {
                ::printf("Found: Wd %s\n", Wd.GetBase16().c_str());
            }
            return false;

        }

    }

    return true;

}

// ----------------------------------------------------------------------------

bool Kangaroo::AddToTable(Int* pos, Int* dist, uint32_t kType) {

    int addStatus = hashTable.Add(pos, dist, kType);
    if (addStatus == ADD_COLLISION)
        return CollisionCheck(&hashTable.kDist, hashTable.kType, dist, kType);

    return addStatus == ADD_OK;

}

bool Kangaroo::AddToTable(uint64_t h, int128_t* x, int128_t* d) {

    int addStatus = hashTable.Add(h, x, d);
    if (addStatus == ADD_COLLISION) {

        Int dist;
        uint32_t kType;
        HashTable::CalcDistAndType(*d, &dist, &kType);
        return CollisionCheck(&hashTable.kDist, hashTable.kType, &dist, kType);

    }

    return addStatus == ADD_OK;

}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyCPU(TH_PARAM* ph) {

    vector<ITEM> dps;
    double lastSent = 0;

    // Global init
    int thId = ph->threadId;

    // Create Kangaroos
    ph->nbKangaroo = CPU_GRP_SIZE;

#ifdef USE_SYMMETRY
    ph->symClass = new uint64_t[CPU_GRP_SIZE];
    for (int i = 0; i < CPU_GRP_SIZE; i++) ph->symClass[i] = 0;
#endif

    IntGroup* grp = new IntGroup(CPU_GRP_SIZE);
    Int* dx = new Int[CPU_GRP_SIZE];

    if (ph->px == NULL) {

        // Create Kangaroos, if not already loaded
        ph->px = new Int[CPU_GRP_SIZE];
        ph->py = new Int[CPU_GRP_SIZE];
        ph->distance = new Int[CPU_GRP_SIZE];
        ph->startDistance = NULL;
        CreateHerd(CPU_GRP_SIZE, ph->px, ph->py, ph->distance, TAME);

    }

    // Allocate and initialize startDistance for wilds mode
    if (wildsOnlyMode) {
        if (ph->startDistance == NULL) {
            ph->startDistance = new Int[CPU_GRP_SIZE];
            for (int g = 0; g < CPU_GRP_SIZE; g++) {
                ph->startDistance[g].Set(&ph->distance[g]);
            }
        }
    }

    if (keyIdx == 0)
        ::printf("SolveKeyCPU Thread %d: %d kangaroos\n", ph->threadId, CPU_GRP_SIZE);

    ph->hasStarted = true;

    // Using Affine coord
    Int dy;
    Int rx;
    Int ry;
    Int _s;
    Int _p;

    while (!endOfSearch) {

        // Random walk

        for (int g = 0; g < CPU_GRP_SIZE; g++) {

#ifdef USE_SYMMETRY
            uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP / 2) + (NB_JUMP / 2) * ph->symClass[g];
#else
            uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif

            Int* p1x = &jumpPointx[jmp];
            Int* p2x = &ph->px[g];
            dx[g].ModSub(p2x, p1x);

        }

        grp->Set(dx);
        grp->ModInv();

        for (int g = 0; g < CPU_GRP_SIZE; g++) {

#ifdef USE_SYMMETRY
            uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP / 2) + (NB_JUMP / 2) * ph->symClass[g];
#else
            uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif

            Int* p1x = &jumpPointx[jmp];
            Int* p1y = &jumpPointy[jmp];
            Int* p2x = &ph->px[g];
            Int* p2y = &ph->py[g];

            dy.ModSub(p2y, p1y);
            _s.ModMulK1(&dy, &dx[g]);
            _p.ModSquareK1(&_s);

            rx.ModSub(&_p, p1x);
            rx.ModSub(p2x);

            ry.ModSub(p2x, &rx);
            ry.ModMulK1(&_s);
            ry.ModSub(p2y);

            ph->distance[g].ModAddK1order(&jumpDistance[jmp]);

#ifdef USE_SYMMETRY
            // Equivalence symmetry class switch
            if (ry.ModPositiveK1()) {
                ph->distance[g].ModNegK1order();
                ph->symClass[g] = !ph->symClass[g];
            }
#endif

            ph->px[g].Set(&rx);
            ph->py[g].Set(&ry);

        }

        if (clientMode) {

            // Send DP to server
            for (int g = 0; g < CPU_GRP_SIZE; g++) {
                if (IsDP(ph->px[g].bits64[3])) {
                    ITEM it;
                    it.x.Set(&ph->px[g]);
                    it.d.Set(&ph->distance[g]);
                    it.kIdx = g;
                    dps.push_back(it);
                }
            }

            double now = Timer::get_tick();
            if (now - lastSent > SEND_PERIOD) {
                LOCK(ghMutex);
                SendToServer(dps, ph->threadId, 0xFFFF);
                UNLOCK(ghMutex);
                lastSent = now;
            }

            if (!endOfSearch) counters[thId] += CPU_GRP_SIZE;

        }
        else {

            // Add to table and collision check
            for (int g = 0; g < CPU_GRP_SIZE && !endOfSearch; g++) {

                if (IsDP(ph->px[g].bits64[3])) {
                    LOCK(ghMutex);
                    if (!endOfSearch) {
                        uint32_t kType;
                        if (wildsOnlyMode)
                            kType = WILD;
                        else if (_generateTameMode)
                            kType = TAME;
                        else
                            kType = (g % 2);
                        if (wildsOnlyMode) {
                            // Write the DP to file
                            WriteWildsDP(&ph->px[g], &ph->distance[g], &ph->startDistance[g]);
                        }
                        if (!AddToTable(&ph->px[g], &ph->distance[g], kType)) {
                            if (wildsOnlyMode) {
                                if (wildsSubRangeMode) {
                                    // Collision inside the same herd in --wilds subrange mode:
                                    // Only reset the kangaroo that found the DP, with a fresh
                                    // random start in the current subrange. The rest of the
                                    // herd keeps walking undisturbed.
                                    Int newStart;
                                    NextWildsSubRangeStart(newStart);

                                    Point newPub = secp->ComputePublicKey(&newStart);
                                    newPub.y.ModNeg();
                                    Point newPos = secp->AddDirect(keyToSearch, newPub);
                                    ph->px[g].Set(&newPos.x);
                                    ph->py[g].Set(&newPos.y);
                                    ph->distance[g].Set(&newStart);
                                    ph->startDistance[g].Set(&newStart);
                                }
                                else {
                                    // Collision inside the same herd in --wilds mode:
                                    // Pick a fresh random sequential base and reinitialize
                                    // ALL kangaroos in this thread's herd from scratch.
                                    wildsNextStart.Rand(wildsOffsetBits);
                                    if (wildsNextStart.IsZero()) wildsNextStart.SetInt32(1);
                                    ::printf("\n[Wilds] Same-herd collision — new sequential base: %s\n",
                                        wildsNextStart.GetBase16().c_str());

                                    // Reinitialize every kangaroo in this herd
                                    for (int r = 0; r < CPU_GRP_SIZE; r++) {
                                        Int newStart;
                                        newStart.Set(&wildsNextStart);
                                        wildsNextStart.Add(&wildsStep);

                                        Point newPub = secp->ComputePublicKey(&newStart);
                                        newPub.y.ModNeg();
                                        Point newPos = secp->AddDirect(keyToSearch, newPub);
                                        ph->px[r].Set(&newPos.x);
                                        ph->py[r].Set(&newPos.y);
                                        ph->distance[r].Set(&newStart);
                                        ph->startDistance[r].Set(&newStart);
                                    }
                                    // Break out of the inner DP loop; the outer while loop
                                    // will restart the walk with the fresh herd.
                                    collisionInSameHerd++;
                                    break;
                                }
                            }
                            else {
                                // Reset kangaroo (same herd collision)
                                CreateHerd(1, &ph->px[g], &ph->py[g], &ph->distance[g],
                                    g % 2, false);
                            }
                            collisionInSameHerd++;
                        }
                        else if (wildsOnlyMode) {
                            // DP successfully stored — reset this kangaroo to its next start
                            Int newStart;
                            if (wildsSubRangeMode)
                                NextWildsSubRangeStart(newStart);
                            else {
                                newStart.Set(&wildsNextStart);
                                wildsNextStart.Add(&wildsStep);
                            }

                            Point newPub = secp->ComputePublicKey(&newStart);
                            newPub.y.ModNeg();
                            Point newPos = secp->AddDirect(keyToSearch, newPub);
                            ph->px[g].Set(&newPos.x);
                            ph->py[g].Set(&newPos.y);
                            ph->distance[g].Set(&newStart);
                            ph->startDistance[g].Set(&newStart);
                        }
                    }
                    UNLOCK(ghMutex);
                }

                if (!endOfSearch) counters[thId]++;

            }

        }

        // Save request
        if (saveRequest && !endOfSearch) {
            ph->isWaiting = true;
            LOCK(saveMutex);
            ph->isWaiting = false;
            UNLOCK(saveMutex);
        }

    }

    // Free
    delete grp;
    delete[] dx;
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
    safe_delete_array(ph->startDistance);
#ifdef USE_SYMMETRY
    safe_delete_array(ph->symClass);
#endif

    ph->isRunning = false;

}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyGPU(TH_PARAM* ph) {

    double lastSent = 0;
    int thId = ph->threadId;

#ifdef WITHGPU

    vector<ITEM> dps;
    vector<ITEM> gpuFound;
    GPUEngine* gpu;

    gpu = new GPUEngine(ph->gridSizeX, ph->gridSizeY, ph->gpuId, 65536 * 2);

    if (keyIdx == 0)
        ::printf("GPU: %s (%.1f MB used)\n", gpu->deviceName.c_str(), gpu->GetMemory() / 1048576.0);

    gpu->SetWildsMode(wildsOnlyMode);

    double t0 = Timer::get_tick();

    if (ph->px == NULL) {
        if (keyIdx == 0)
            ::printf("SolveKeyGPU Thread GPU#%d: creating kangaroos...\n", ph->gpuId);
        uint64_t nbThread = gpu->GetNbThread();
        ph->px = new Int[ph->nbKangaroo];
        ph->py = new Int[ph->nbKangaroo];
        ph->distance = new Int[ph->nbKangaroo];
        ph->startDistance = NULL;

        for (uint64_t i = 0; i < nbThread; i++) {
            CreateHerd(GPU_GRP_SIZE, &(ph->px[i * GPU_GRP_SIZE]),
                &(ph->py[i * GPU_GRP_SIZE]),
                &(ph->distance[i * GPU_GRP_SIZE]),
                TAME);
        }
    }

    // Allocate and initialize startDistance for wilds mode
    if (wildsOnlyMode) {
        if (ph->startDistance == NULL) {
            ph->startDistance = new Int[ph->nbKangaroo];
            for (uint64_t g = 0; g < ph->nbKangaroo; g++) {
                ph->startDistance[g].Set(&ph->distance[g]);
            }
        }
    }

#ifdef USE_SYMMETRY
    if (!wildsOnlyMode)
        gpu->SetWildOffset(&rangeWidthDiv4);
    else
        gpu->SetWildOffset(nullptr);
#else
    if (!wildsOnlyMode)
        gpu->SetWildOffset(&rangeWidthDiv2);
    else
        gpu->SetWildOffset(nullptr);
#endif

    gpu->SetParams(dMask, jumpDistance, jumpPointx, jumpPointy);
    gpu->SetKangaroos(ph->px, ph->py, ph->distance);

    if (workFile.length() == 0 || !saveKangaroo) {
        safe_delete_array(ph->px);
        safe_delete_array(ph->py);
        safe_delete_array(ph->distance);
    }

    if (!gpu->callKernel()) {
        printf("GPU: callKernel failed\n");
        delete gpu;
        ph->isRunning = false;
        return;
    }

    double t1 = Timer::get_tick();

    if (keyIdx == 0)
        ::printf("SolveKeyGPU Thread GPU#%d: 2^%.2f kangaroos [%.1fs]\n", ph->gpuId, log2((double)ph->nbKangaroo), (t1 - t0));

    ph->hasStarted = true;

    while (!endOfSearch) {

        auto _kernelT0 = std::chrono::high_resolution_clock::now();
        if (!gpu->Launch(gpuFound)) {
            printf("GPU: Launch failed\n");
            break;
        }
        auto _kernelT1 = std::chrono::high_resolution_clock::now();
        lastKernelMs = std::chrono::duration<double, std::milli>(_kernelT1 - _kernelT0).count();
        counters[thId] += ph->nbKangaroo * NB_RUN;

        if (clientMode) {
            for (int i = 0; i < (int)gpuFound.size(); i++)
                dps.push_back(gpuFound[i]);

            double now = Timer::get_tick();
            if (now - lastSent > SEND_PERIOD) {
                LOCK(ghMutex);
                SendToServer(dps, ph->threadId, ph->gpuId);
                UNLOCK(ghMutex);
                lastSent = now;
            }
        }
        else {
            if (gpuFound.size() > 0) {
                LOCK(ghMutex);
                for (int g = 0; !endOfSearch && g < (int)gpuFound.size(); g++) {
                    uint32_t kType = wildsOnlyMode ? WILD : (uint32_t)(gpuFound[g].kIdx % 2);
                    PrintDPDebug(&gpuFound[g].x, &gpuFound[g].d, kType, thId, "GPU");

                    if (wildsOnlyMode) {
                        WriteWildsDP(&gpuFound[g].x, &gpuFound[g].d, &ph->startDistance[gpuFound[g].kIdx]);
                    }

                    if (!AddToTable(&gpuFound[g].x, &gpuFound[g].d, kType)) {
                        if (wildsOnlyMode) {
                            if (wildsSubRangeMode) {
                                // Collision inside the same herd in --wilds subrange mode:
                                // Only reset the kangaroo that found the DP, with a fresh
                                // random start in the current subrange. The rest of the
                                // herd keeps walking undisturbed.
                                Int newStart;
                                NextWildsSubRangeStart(newStart);

                                Point newPub = secp->ComputePublicKey(&newStart);
                                newPub.y.ModNeg();
                                Point newPos = secp->AddDirect(keyToSearch, newPub);
                                Int npx, npy;
                                npx.Set(&newPos.x);
                                npy.Set(&newPos.y);
                                gpu->SetKangaroo(gpuFound[g].kIdx, &npx, &npy, &newStart);
                                ph->startDistance[gpuFound[g].kIdx].Set(&newStart);
                            }
                            else {
                                // Collision inside the same herd in --wilds mode:
                                // Pick a fresh random sequential base and reinitialize
                                // ALL kangaroos on this GPU from scratch.
                                wildsNextStart.Rand(wildsOffsetBits);
                                if (wildsNextStart.IsZero()) wildsNextStart.SetInt32(1);
                                ::printf("\n[Wilds GPU] Same-herd collision — new sequential base: %s\n",
                                    wildsNextStart.GetBase16().c_str());

                                // Reinitialize every kangaroo on this GPU
                                for (uint64_t r = 0; r < ph->nbKangaroo; r++) {
                                    Int newStart;
                                    newStart.Set(&wildsNextStart);
                                    wildsNextStart.Add(&wildsStep);

                                    Point newPub = secp->ComputePublicKey(&newStart);
                                    newPub.y.ModNeg();
                                    Point newPos = secp->AddDirect(keyToSearch, newPub);
                                    Int npx, npy;
                                    npx.Set(&newPos.x);
                                    npy.Set(&newPos.y);
                                    gpu->SetKangaroo(r, &npx, &npy, &newStart);
                                    ph->startDistance[r].Set(&newStart);
                                }
                                // Break out of the gpuFound loop; the outer while loop
                                // will restart the walk with the fresh herd.
                                collisionInSameHerd++;
                                break;
                            }
                        }
                        else {
                            Int px, py, d;
                            CreateHerd(1, &px, &py, &d, kType, false);
                            gpu->SetKangaroo(gpuFound[g].kIdx, &px, &py, &d);
                        }
                        collisionInSameHerd++;
                    }
                    else if (wildsOnlyMode) {
                        // DP successfully stored — reset this kangaroo to its next start
                        Int newStart;
                        if (wildsSubRangeMode)
                            NextWildsSubRangeStart(newStart);
                        else {
                            newStart.Set(&wildsNextStart);
                            wildsNextStart.Add(&wildsStep);
                        }

                        Point newPub = secp->ComputePublicKey(&newStart);
                        newPub.y.ModNeg();
                        Point newPos = secp->AddDirect(keyToSearch, newPub);
                        Int npx, npy;
                        npx.Set(&newPos.x);
                        npy.Set(&newPos.y);
                        gpu->SetKangaroo(gpuFound[g].kIdx, &npx, &npy, &newStart);
                        ph->startDistance[gpuFound[g].kIdx].Set(&newStart);
                    }
                }
                UNLOCK(ghMutex);
            }
        }

        if (saveRequest && !endOfSearch) {
            if (saveKangaroo)
                gpu->GetKangaroos(ph->px, ph->py, ph->distance);
            ph->isWaiting = true;
            LOCK(saveMutex);
            ph->isWaiting = false;
            UNLOCK(saveMutex);
        }
    }

    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
    safe_delete_array(ph->startDistance);
    delete gpu;

#else

    ph->hasStarted = true;

#endif

    ph->isRunning = false;
}
// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _SolveKeyCPU(LPVOID lpParam) {
#else
void* _SolveKeyCPU(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->obj->SolveKeyCPU(p);
    return 0;
}

#ifdef WIN64
DWORD WINAPI _SolveKeyGPU(LPVOID lpParam) {
#else
void* _SolveKeyGPU(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->obj->SolveKeyGPU(p);
    return 0;
}

// ----------------------------------------------------------------------------

void Kangaroo::CreateHerd(int nbKangaroo, Int * px, Int * py, Int * d, int firstType, bool lock) {

    vector<Int> pk;
    vector<Point> S;
    vector<Point> Sp;
    pk.reserve(nbKangaroo);
    S.reserve(nbKangaroo);
    Sp.reserve(nbKangaroo);
    Point Z;
    Z.Clear();

    if (lock) LOCK(ghMutex);

    for (uint64_t j = 0; j < (uint64_t)nbKangaroo; ++j) {

        if (wildsOnlyMode) {
            if (wildsSubRangeMode) {
                // Each kangaroo gets a random start within its own subrange;
                // the subrange tracker is advanced for every kangaroo created.
                NextWildsSubRangeStart(d[j]);
            }
            else {
                // Sequential starting points: base + j * 2^stepBits, tracked via wildsNextStart
                // wildsNextStart is advanced externally before each CreateHerd call (bulk init)
                // or inline here for single-kangaroo resets.
                // For bulk creation: assign wildsNextStart then advance it.
                d[j].Set(&wildsNextStart);
                wildsNextStart.Add(&wildsStep);   // advance tracker for next kangaroo
            }
        }
        else {
            // Original behaviour
#ifdef USE_SYMMETRY
            d[j].Rand(rangePower - 1);
            if ((j + firstType) % 2 == WILD) {
                d[j].ModSubK1order(&rangeWidthDiv4);
            }
#else
            d[j].Rand(rangePower);
            if (!_generateTameMode && ((j + firstType) % 2 == WILD)) {
                d[j].ModSubK1order(&rangeWidthDiv2);
            }
#endif
        }
        pk.push_back(d[j]);
    }

    if (lock) UNLOCK(ghMutex);

    // --- MODIFIED SECTION ---
    if (wildsOnlyMode) {
        // Iterate through each private key and compute public keys individually
        for (uint64_t j = 0; j < (uint64_t)nbKangaroo; ++j) {
            Point singlePoint = secp->ComputePublicKey(&pk[j]);
            singlePoint.y.ModNeg();
            S.push_back(singlePoint);
        }
    }
    else {
        // Use standard batch computation for normal mode
        S = secp->ComputePublicKeys(pk);
    }
    // ------------------------

    for (uint64_t j = 0; j < (uint64_t)nbKangaroo; ++j) {
        if (wildsOnlyMode) {
            Sp.push_back(keyToSearch);
        }
        else {
            if ((j + firstType) % 2 == TAME)
                Sp.push_back(Z);
            else
                Sp.push_back(keyToSearch);
        }
    }

    S = secp->AddDirect(Sp, S);

    for (uint64_t j = 0; j < (uint64_t)nbKangaroo; ++j) {
        px[j].Set(&S[j].x);
        py[j].Set(&S[j].y);

        if (!wildsOnlyMode) {
#ifdef USE_SYMMETRY
            if (py[j].ModPositiveK1())
                d[j].ModNegK1order();
#endif
        }
    }
}

// ----------------------------------------------------------------------------
/*
void Kangaroo::CreateJumpTable() {

#ifdef USE_SYMMETRY
  int jumpBit = rangePower / 2;
#else
  int jumpBit = rangePower / 2 + 1;
#endif

  if(jumpBit > 128) jumpBit = 128;
  int maxRetry = 100;
  bool ok = false;
  double distAvg;
  double maxAvg = pow(2.0,(double)jumpBit - 0.95);
  double minAvg = pow(2.0,(double)jumpBit - 1.05);
  //::printf("Jump Avg distance min: 2^%.2f\n",log2(minAvg));
  //::printf("Jump Avg distance max: 2^%.2f\n",log2(maxAvg));

  // Kangaroo jumps
  // Constant seed for compatibilty of workfiles
  rseed(0x600DCAFE);

#ifdef USE_SYMMETRY
  Int old;
  old.Set(Int::GetFieldCharacteristic());
  Int u;
  Int v;
  u.SetInt32(1);
  u.ShiftL(jumpBit/2);
  u.AddOne();
  while(!u.IsProbablePrime()) {
    u.AddOne();
    u.AddOne();
  }
  v.Set(&u);
  v.AddOne();
  v.AddOne();
  while(!v.IsProbablePrime()) {
    v.AddOne();
    v.AddOne();
  }
  Int::SetupField(&old);

  ::printf("U= %s\n",u.GetBase16().c_str());
  ::printf("V= %s\n",v.GetBase16().c_str());
#endif

  // Positive only
  // When using symmetry, the sign is switched by the symmetry class switch
  while(!ok && maxRetry>0 ) {
    Int totalDist;
    totalDist.SetInt32(0);
#ifdef USE_SYMMETRY
    for(int i = 0; i < NB_JUMP/2; ++i) {
      jumpDistance[i].Rand(jumpBit/2);
      jumpDistance[i].Mult(&u);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
    }
    for(int i = NB_JUMP / 2; i < NB_JUMP; ++i) {
      jumpDistance[i].Rand(jumpBit/2);
      jumpDistance[i].Mult(&v);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
    }
#else
    for(int i = 0; i < NB_JUMP; ++i) {
      jumpDistance[i].Rand(jumpBit);
      if(jumpDistance[i].IsZero())
        jumpDistance[i].SetInt32(1);
      totalDist.Add(&jumpDistance[i]);
  }
#endif
    distAvg = totalDist.ToDouble() / (double)(NB_JUMP);
    ok = distAvg>minAvg && distAvg<maxAvg;
    maxRetry--;
  }

  for(int i = 0; i < NB_JUMP; ++i) {
    Point J = secp->ComputePublicKey(&jumpDistance[i]);
    jumpPointx[i].Set(&J.x);
    jumpPointy[i].Set(&J.y);
  }

  ::printf("Jump Avg distance: 2^%.2f\n",log2(distAvg));

  unsigned long seed = Timer::getSeed32();
  rseed(seed);

}*/
void Kangaroo::CreateJumpTable() {
    // Override NB_JUMP logic locally or ensure NB_JUMP is defined as 0x100 (256)
    const int totalEntries = 0x100;

    // Initialize Fibonacci sequence variables
    uint64_t fib1 = 0;
    uint64_t fib2 = 1;

    int i = 0;

    // 1. Fill the first entries with Fibonacci numbers less than 2,000,000
    // Note: Starting from 1 (the first non-zero Fibonacci result of fib1 + fib2)
    while (i < totalEntries) {
        uint64_t nextFib = fib1 + fib2;
        if (nextFib >= 2000000) {
            break; // Stop if the next Fibonacci number hits or exceeds 2,000,000
        }

        jumpDistance[i].SetInt32(nextFib);

        // Move to next Fibonacci numbers
        fib1 = fib2;
        fib2 = nextFib;
        i++;
    }

    // 2. Fill the remaining entries up to 0xFF (256 total) with 1
    while (i < totalEntries) {
        jumpDistance[i].SetInt32(1);
        i++;
    }

    // 3. Compute the EC public key points for all 0x100 entries
    for (int j = 0; j < totalEntries; ++j) {
        Point J = secp->ComputePublicKey(&jumpDistance[j]);
        J.y.ModNeg();
        jumpPointx[j].Set(&J.x);
        jumpPointy[j].Set(&J.y);
    }

    // Re-seed the RNG as the original code did to maintain external compatibility
    unsigned long seed = Timer::getSeed32();
    rseed(seed);
}

// ----------------------------------------------------------------------------

void Kangaroo::ComputeExpected(double dp, double* op, double* ram, double* overHead) {

    // Compute expected number of operation and memory

#ifdef USE_SYMMETRY
    double gainS = 1.0 / sqrt(2.0);
#else
    double gainS = 1.0;
#endif

    // Kangaroo number
    double k = (double)totalRW;

    // Range size
    double N = pow(2.0, (double)rangePower);

    // theta
    double theta = pow(2.0, dp);

    // Z0
    double Z0 = (2.0 * (2.0 - sqrt(2.0)) * gainS) * sqrt(M_PI);

    // Average for DP = 0
    double avgDP0 = Z0 * sqrt(N);

    // DP Overhead
    *op = Z0 * pow(N * (k * theta + sqrt(N)), 1.0 / 3.0);

    *ram = (double)sizeof(HASH_ENTRY) * (double)HASH_SIZE + // Table
        (double)sizeof(ENTRY*) * (double)(HASH_SIZE * 4) + // Allocation overhead
        (double)(sizeof(ENTRY) + sizeof(ENTRY*)) * (*op / theta); // Entries

    *ram /= (1024.0 * 1024.0);

    if (overHead)
        *overHead = *op / avgDP0;

}

// ----------------------------------------------------------------------------

void Kangaroo::InitRange() {

    rangeWidth.Set(&rangeEnd);
    rangeWidth.Sub(&rangeStart);
    rangePower = rangeWidth.GetBitLength();
    ::printf("Range width: 2^%d\n", rangePower);
    rangeWidthDiv2.Set(&rangeWidth);
    rangeWidthDiv2.ShiftR(1);
    rangeWidthDiv4.Set(&rangeWidthDiv2);
    rangeWidthDiv4.ShiftR(1);
    rangeWidthDiv8.Set(&rangeWidthDiv4);
    rangeWidthDiv8.ShiftR(1);

}

void Kangaroo::InitSearchKey() {

    Int SP;
    SP.Set(&rangeStart);
#ifdef USE_SYMMETRY
    SP.ModAddK1order(&rangeWidthDiv2);
#endif
    if (!SP.IsZero()) {
        Point RS = secp->ComputePublicKey(&SP);
        RS.y.ModNeg();
        keyToSearch = secp->AddDirect(keysToSearch[keyIdx], RS);
    }
    else {
        keyToSearch = keysToSearch[keyIdx];
    }
    keyToSearchNeg = keyToSearch;
    keyToSearchNeg.y.ModNeg();

}

// -----------------------

void Kangaroo::PrintStatus() {
    // Sum all jumps (counters is uint64_t[256])
    uint64_t totalJumps = 0;
    for (int i = 0; i < 256; ++i) {
        totalJumps += counters[i - 1];
    }

    double elapsed = Timer::get_tick() - startTime;
    if (elapsed < 0.001) elapsed = 0.001;
    double speed = (double)totalJumps / elapsed / 1e6;   // mega jumps/sec

    int hours = (int)(elapsed / 3600);
    int mins = (int)((elapsed - hours * 3600) / 60);
    int secs = (int)(elapsed - hours * 3600 - mins * 60);

    // Choose a thread to display its jumps (in hex)
    int displayThread = -1;
    uint64_t threadJumps = 0;


    if (nbCPUThread > 0) {
        displayThread = 0;                     // first CPU thread
        threadJumps = counters[0];
    }
    else if (nbGPUThread > 0) {
        displayThread = nbCPUThread;           // first GPU thread (ID 128)
        threadJumps = counters[displayThread];
    }

    // Print status line with hex values
    if (displayThread >= 0) {
        printf("\rSpeed: %.2f MJ/s | (#%d): %" PRIX64 " | Total: %" PRIX64 " | Resets: %d | Time: %d:%02d:%02d | Kernel: %.1fms   ",
            speed, displayThread, threadJumps, totalJumps, collisionInSameHerd, hours, mins, secs, lastKernelMs);
    }
    else {
        // No threads? Should not happen
        printf("\rSpeed: %.2f MJ/s | Total: %" PRIX64 " | Resets: %d | Time: %d:%02d:%02d | Kernel: %.1fms   ",
            speed, totalJumps, collisionInSameHerd, hours, mins, secs, lastKernelMs);
    }
    fflush(stdout);
}

// ----------------------------------------------------------------------------

void Kangaroo::Run(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize) {

    double t0 = Timer::get_tick();
    startTime = t0;

    nbCPUThread = nbThread;
    nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
    totalRW = 0;

#ifndef WITHGPU
    if (nbGPUThread > 0) {
        ::printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
        nbGPUThread = 0;
    }
#endif

    // Disable GPU if wilds-only mode is on and symmetry is still defined
    if (wildsOnlyMode && nbGPUThread > 0) {
#ifdef USE_SYMMETRY
        ::printf("Wilds-only GPU requires compilation without -DUSE_SYMMETRY. Falling back to CPU only.\n");
        nbGPUThread = 0;
#else
        ::printf("Wilds-only mode with GPU (symmetry disabled)\n");
#endif
    }

    uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
    if (totalThread == 0) {
        ::printf("No CPU or GPU thread, exiting.\n");
        ::exit(0);
    }

    TH_PARAM* params = (TH_PARAM*)malloc(totalThread * sizeof(TH_PARAM));
    THREAD_HANDLE* thHandles = (THREAD_HANDLE*)malloc(totalThread * sizeof(THREAD_HANDLE));

    memset(params, 0, totalThread * sizeof(TH_PARAM));
    memset(counters, 0, sizeof(counters));
    ::printf("Number of CPU thread: %d\n", nbCPUThread);

#ifdef WITHGPU
    for (int i = 0; i < nbGPUThread; i++) {
        int x = gridSize[2ULL * i];
        int y = gridSize[2ULL * i + 1ULL];
        if (!GPUEngine::GetGridSize(gpuId[i], &x, &y)) {
            return;
        }
        else {
            params[nbCPUThread + i].gridSizeX = x;
            params[nbCPUThread + i].gridSizeY = y;
        }
        params[nbCPUThread + i].nbKangaroo = (uint64_t)GPU_GRP_SIZE * x * y;
        totalRW += params[nbCPUThread + i].nbKangaroo;
    }
#endif

    totalRW += nbCPUThread * (uint64_t)CPU_GRP_SIZE;

    if (clientMode) {
        if (!GetConfigFromServer())
            ::exit(0);
        if (workFile.length() > 0)
            saveKangaroo = true;
    }

    // Wilds-only initialisation
    if (wildsOnlyMode) {
        rangePower = wildsOffsetBits + 1;
        keyIdx = 0;
        rangeStart.SetInt32(0);
        InitSearchKey();
        ::printf("Wilds only mode, offset bits: %d\n", wildsOffsetBits);

        if (wildsSubRangeMode) {

            // Split [1, 2^offsetBits] into totalRW equal subranges (one per kangaroo).
            // Round the kangaroo count up to the next power of two so each subrange
            // is an exact power-of-two slice (allows uniform sampling via Rand()).
            int subRangeBits = 0;
            while (((uint64_t)1 << subRangeBits) < totalRW)
                subRangeBits++;
            wildsNbSubRanges = (uint64_t)1 << subRangeBits;

            wildsSubRangeBits = wildsOffsetBits - subRangeBits;
            if (wildsSubRangeBits < 0) wildsSubRangeBits = 0;

            wildsSubRangeIdx = 0;

            ::printf("Wilds subrange mode: %llu subranges of 2^%d each\n",
                (unsigned long long)wildsNbSubRanges, wildsSubRangeBits);
        }
        else {
            /*
            // Build the 2^35 step constant
            wildsStep.SetInt32(1);
            wildsStep.ShiftL(38);          // wildsStep = 2^35
            */

            // Pick a random base in [0, 2^wildsOffsetBits) as the first starting point
            wildsNextStart.Rand(wildsOffsetBits);
            if (wildsNextStart.IsZero()) wildsNextStart.SetInt32(1);
            ::printf("Wilds sequential base: %s\n", wildsNextStart.GetBase16().c_str());
        }
    }
    else {
        InitRange();
    }

    CreateJumpTable();

    ::printf("Number of kangaroos: 2^%.2f\n", log2((double)totalRW));

    if (!clientMode && !wildsOnlyMode) {
        double dpOverHead;
        int suggestedDP = (int)((double)rangePower / 2.0 - log2((double)totalRW));
        if (suggestedDP < 0) suggestedDP = 0;
        ComputeExpected((double)suggestedDP, &expectedNbOp, &expectedMem, &dpOverHead);
        while (dpOverHead > 1.05 && suggestedDP > 0) {
            suggestedDP--;
            ComputeExpected((double)suggestedDP, &expectedNbOp, &expectedMem, &dpOverHead);
        }
        if (initDPSize < 0)
            initDPSize = suggestedDP;
        ComputeExpected((double)initDPSize, &expectedNbOp, &expectedMem);
        if (nbLoadedWalk == 0) ::printf("Suggested DP: %d\n", suggestedDP);
        ::printf("Expected operations: 2^%.2f\n", log2(expectedNbOp));
        ::printf("Expected RAM: %.1fMB\n", expectedMem);
    }
    else if (wildsOnlyMode) {
        if (initDPSize < 0) initDPSize = 20;
        ::printf("Wilds mode DP size: %d\n", initDPSize);
    }

    SetDP(initDPSize);
    FectchKangaroos(params);

    for (keyIdx = 0; keyIdx < (int)keysToSearch.size(); keyIdx++) {

        if (!wildsOnlyMode)
            InitSearchKey();

        endOfSearch = false;
        collisionInSameHerd = 0;
        memset(counters, 0, sizeof(counters));

        // Launch CPU threads
        for (int i = 0; i < nbCPUThread; i++) {
            params[i].threadId = i;
            params[i].isRunning = true;
            thHandles[i] = LaunchThread(_SolveKeyCPU, params + i);
        }

#ifdef WITHGPU
        for (int i = 0; i < nbGPUThread; i++) {
            int id = nbCPUThread + i;
            params[id].threadId = 0x80L + i;
            params[id].isRunning = true;
            params[id].gpuId = gpuId[i];
            thHandles[id] = LaunchThread(_SolveKeyGPU, params + id);
        }
#endif

        double lastSaveTime = Timer::get_tick();
        int saveCounter = 0;
        bool threadsRunning = true;

        while (threadsRunning) {
            threadsRunning = false;
            for (uint64_t i = 0; i < totalThread; i++) {
                if (params[i].isRunning) {
                    threadsRunning = true;
                    break;
                }
            }

            double nowStatus = Timer::get_tick();
            if (nowStatus - lastStatusTime >= 2.0) {
                PrintStatus();
                lastStatusTime = nowStatus;
            }

            if (_generateTameMode && _generateTameSavePeriod > 0) {
                double now = Timer::get_tick();
                if (now - lastSaveTime > _generateTameSavePeriod) {
                    std::string partFileName = _generateTameFile + ".part" + std::to_string(saveCounter);
                    printf("\nPeriodically saving snapshot to %s and flushing database...\n", partFileName.c_str());
                    std::string originalWorkFile = this->workFile;
                    this->workFile = partFileName;
                    uint64_t currentCount = getCPUCount() + getGPUCount();
                    double currentTime = Timer::get_tick() - t0 + offsetTime;
                    SaveWork(currentCount, currentTime, params, totalThread);
                    this->workFile = originalWorkFile;
                    lastSaveTime = now;
                    saveCounter++;
                }
            }

            if (threadsRunning) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        JoinThreads(thHandles, nbCPUThread + nbGPUThread);
        FreeHandles(thHandles, nbCPUThread + nbGPUThread);

        if (_generateTameMode) {
            std::string partFileName = _generateTameFile + ".part" + std::to_string(saveCounter);
            printf("\nFinal snapshot save to %s and flushing database...\n", partFileName.c_str());
            std::string originalWorkFile = this->workFile;
            this->workFile = partFileName;
            uint64_t currentCount = getCPUCount() + getGPUCount();
            double currentTime = Timer::get_tick() - t0 + offsetTime;
            SaveWork(currentCount, currentTime, params, totalThread);
            this->workFile = originalWorkFile;
        }

        hashTable.Reset();
    }

    double t1 = Timer::get_tick();
    ::printf("\nDone: Total time %s \n", GetTimeStr(t1 - t0 + offsetTime).c_str());
}

void Kangaroo::SetWildsMode(int offsetBits, int stepBits, const std::string & pubKeyHex) {
    wildsOnlyMode = true;
    wildsOffsetBits = offsetBits;
    wildsStepBits = stepBits;
    wildsSubRangeMode = (stepBits == 0);

    if (!wildsSubRangeMode) {
        // Build the step = 2^stepBits
        wildsStep.SetInt32(1);
        wildsStep.ShiftL(stepBits);   // 2^stepBits
    }

    if (!pubKeyHex.empty()) {
        Point p;
        bool isCompressed;
        if (!secp->ParsePublicKeyHex(pubKeyHex, p, isCompressed)) {
            printf("Invalid public key: %s\n", pubKeyHex.c_str());
            exit(-1);
        }
        keysToSearch.clear();
        keysToSearch.push_back(p);
    }
}

void Kangaroo::GenerateWildDistance(Int & dist) {
    dist.Rand(wildsOffsetBits);
    if (dist.IsZero())
        dist.SetInt32(1);   // ensure range 1 .. 2^bits
}

// ----------------------------------------------------------------------------
// Subrange mode (--wilds offsetBits,0)
//
// The range [1, 2^offsetBits] is split into wildsNbSubRanges equal subranges
// (one per kangaroo, across all CPU/GPU threads), each of size 2^wildsSubRangeBits.
// wildsNbSubRanges is rounded up to the next power of two so that each
// subrange is an exact power-of-two-sized slice; this lets us pick a uniform
// random point inside a subrange with a single Rand() call (no modulo needed).
//
// wildsSubRangeIdx tracks the next subrange to hand out, round-robin, wrapping
// back to 0 once every subrange has been used. Each call picks a uniformly
// random point inside the current subrange, then advances the tracker to the
// next subrange. Must be called under ghMutex (same locking discipline as
// wildsNextStart).
// ----------------------------------------------------------------------------
void Kangaroo::NextWildsSubRangeStart(Int & start) {

    // subRangeStart = 1 + idx * 2^wildsSubRangeBits
    Int offset;
    offset.SetInt32((int32_t)wildsSubRangeIdx);
    offset.ShiftL(wildsSubRangeBits);

    start.SetInt32(1);
    start.Add(&offset);

    // Random offset within the subrange: [0, 2^wildsSubRangeBits)
    if (wildsSubRangeBits > 0) {
        Int randOffset;
        randOffset.Rand(wildsSubRangeBits);
        start.Add(&randOffset);
    }

    if (start.IsZero())
        start.SetInt32(1);

    // Advance to next subrange, wrapping when all subranges have been covered
    wildsSubRangeIdx++;
    if (wildsSubRangeIdx >= wildsNbSubRanges)
        wildsSubRangeIdx = 0;
}

void Kangaroo::WriteWildsDP(Int * x, Int * dist, Int * startDist) {
    // Compute the public key from the distance
    Point PR = secp->ComputePublicKey(dist);
    std::string pubKeyHex = PR.x.GetBase16();
    // Open file, append one line, close (no lock needed)
    FILE* f = fopen(wildsOutputFile.c_str(), "a");
    if (f) {
        fprintf(f, "%s,%s,%s\n", x->GetBase16().c_str(), dist->GetBase16().c_str(), startDist->GetBase16().c_str());
        fclose(f);
    }
    else {
        printf("Cannot open %s for writing\n", wildsOutputFile.c_str());
    }
}