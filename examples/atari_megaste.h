/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef MEGASTE_H
#define MEGASTE_H

#include <stdint.h>

/* $FFFF8E21 Mega STE cache/processor control bits */
#define MEGASTE_CACHE 0x01
#define MEGASTE_16MHZ 0x02

int is_megaste(void);
int megaste_enable_16mhz_cache(void);

/* Read/set the cache/processor control bits (supervisor mode, Mega STE only) */
uint8_t megaste_cpu_mode(void);
void megaste_set_cpu_mode(uint8_t mode);

#endif /* MEGASTE_H */
