/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <stdint.h>
#include <mint/osbind.h>
#include <mint/cookie.h>

#include "atari_megaste.h"

#define MCH_MEGA_STE 0x00010010L

int is_megaste(void)
{
  long mch = 0;
  if (C_FOUND != Getcookie(C__MCH, &mch))
    return 0;
  return mch == MCH_MEGA_STE;
}

int megaste_enable_16mhz_cache(void)
{
  if (is_megaste())
  {
    megaste_set_cpu_mode(MEGASTE_16MHZ | MEGASTE_CACHE);
    return 0;
  }

  return 1;
}

uint8_t megaste_cpu_mode(void)
{
  return *(volatile uint8_t *)0xFFFF8E21 & (MEGASTE_16MHZ | MEGASTE_CACHE);
}

void megaste_set_cpu_mode(uint8_t mode)
{
  *(volatile uint8_t *)0xFFFF8E21 = mode;
}
