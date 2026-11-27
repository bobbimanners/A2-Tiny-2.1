/* TINY linker script: the stock Apple II ProDOS SYS script with the RAM
 * limit moved up.
 *
 * The stock script caps the whole program at $8f00 bytes because that is the
 * largest SYS file ProDOS loads. Only the load image has that limit; zeroed
 * data may sit above it. TINY's soft stack was measured at under 200 bytes
 * (tests/mame-profile.lua, TINY_WM_LO), so 1 KB is kept for it instead of
 * 4 KB and the other 3 KB become usable for code and data.
 */
__rc0 = 0x00;
INCLUDE imag-regs.ld
ASSERT(__rc31 == 0x1f, "Inconsistent zero page map.")

MEMORY {
    zp : ORIGIN = 0x60, LENGTH = 0xa0
    ram (rw) : ORIGIN = 0x2000, LENGTH = 0x9b00
}

REGION_ALIAS("c_readonly", ram)
REGION_ALIAS("c_writeable", ram)

SECTIONS {
    .sys_entry : { KEEP(*(.sys_entry)) } >ram
    INCLUDE c.ld
}
ASSERT(ADDR(.sys_entry) == ORIGIN(ram),
       "ProDOS SYS entry point must be $2000.")
ASSERT(SIZEOF(.sys_entry) == 3,
       "ProDOS SYS entry must be a three-byte JMP.")
ASSERT(__data_end - ORIGIN(ram) <= 0x8f00,
       "Load image exceeds the largest ProDOS SYS file.")

/* The soft stack grows down through the $bb00-$beff area. */
__stack = 0xbf00;

OUTPUT_FORMAT { TRIM(ram) }
