/*
    Copyright 2011, Spyros Blanas.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * CHANGELOG
 *  - changed `unsigned long long' declerations to uint64_t and added include
 *    for <stdint.h>. May 2012, Cagri.
 *
 */
#pragma once

#include <stdint.h>
#include <fstream>

__inline__ double get_cpu_frequency() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            double mhz = std::stod(line.substr(line.find(":") + 1));
            return mhz * 1e6; // MHz to Hz
        }
    }
    return 0.0;
}

const double cpu_frequency = get_cpu_frequency();

__inline__ double to_nano_seconds(uint64_t ticks) {
    return (double) ticks / cpu_frequency * 1e9;
}

__inline__ double to_micro_seconds(uint64_t ticks) {
    return (double) ticks / cpu_frequency * 1e6;
}

__inline__ double to_milli_seconds(uint64_t ticks) {
    return (double) ticks / cpu_frequency * 1e3;
}

__inline__ double to_seconds(uint64_t ticks) {
    return (double) ticks / cpu_frequency;
}

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__i386__) && !defined(__x86_64__) && !defined(__sparc__)
#warning No supported architecture found -- timers will return junk.
#endif


static __inline__ uint64_t curtick() {
    uint64_t tick;
#if defined(__i386__)
    unsigned long lo, hi;
    __asm__ __volatile__ (".byte 0x0f, 0x31" : "=a" (lo), "=d" (hi));
    tick = (uint64_t) hi << 32 | lo;
#elif defined(__x86_64__)
    unsigned long lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    tick = (uint64_t) hi << 32 | lo;
#elif defined(__sparc__)
    __asm__ __volatile__ ("rd %%tick, %0" : "=r" (tick));
#endif
    return tick;
}

static __inline__ void  start_timer(uint64_t *t) {
    *t = curtick();
}

static __inline__ void stop_timer(uint64_t *t) {
    *t = curtick() - *t;
}

static __inline__ void accTimer(uint64_t *pretimer, uint64_t *acctimer) {
    *acctimer += curtick() - *pretimer;
}

static __inline__ void accTimerMulti(uint64_t *pretimer, uint64_t *acctimer, int multiplier) {
    *acctimer += (curtick() - *pretimer) * multiplier;
}
#ifdef __cplusplus
}
#endif