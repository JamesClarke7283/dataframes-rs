//! Shuffle-control tables for vectorised compaction.
//!
//! Compaction ("keep the lanes whose mask bit is set, packed to the front") has
//! no single instruction below AVX-512. With AVX2 it is a table lookup: the
//! mask bits of one vector index a table whose entry is the permutation that
//! moves the selected lanes to the front. Each table is computed at compile
//! time and placed on a cache-line boundary so an entry never straddles lines.

/// Wrapper forcing 64-byte alignment of a table.
#[repr(C, align(64))]
pub struct CacheAligned<T>(pub T);

/// For an 8-bit mask: the 8 lane indices for `vpermd` (32-bit lanes).
/// Selected lanes come first, in order; the rest of the entry is arbitrary.
pub static PERM32: CacheAligned<[[u32; 8]; 256]> = CacheAligned(perm32());

/// For a 4-bit mask over 64-bit lanes: `vpermd` indices moving the selected
/// 64-bit lanes (as pairs of 32-bit halves) to the front.
pub static PERM64: CacheAligned<[[u32; 8]; 16]> = CacheAligned(perm64());

/// For an 8-bit mask over 16-bit lanes: `pshufb` control bytes.
pub static SHUF16: CacheAligned<[[u8; 16]; 256]> = CacheAligned(shuf16());

/// For an 8-bit mask over 8-bit lanes: `pshufb` control bytes (low 8 lanes).
pub static SHUF8: CacheAligned<[[u8; 16]; 256]> = CacheAligned(shuf8());

const fn perm32() -> [[u32; 8]; 256] {
    let mut t = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut i) = (0, 0);
        while i < 8 {
            if m >> i & 1 == 1 {
                t[m][k] = i as u32;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
}

const fn perm64() -> [[u32; 8]; 16] {
    let mut t = [[0u32; 8]; 16];
    let mut m = 0;
    while m < 16 {
        let (mut k, mut i) = (0, 0);
        while i < 4 {
            if m >> i & 1 == 1 {
                t[m][2 * k] = 2 * i as u32;
                t[m][2 * k + 1] = 2 * i as u32 + 1;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
}

const fn shuf16() -> [[u8; 16]; 256] {
    let mut t = [[0u8; 16]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut i) = (0, 0);
        while i < 8 {
            if m >> i & 1 == 1 {
                t[m][2 * k] = 2 * i as u8;
                t[m][2 * k + 1] = 2 * i as u8 + 1;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
}

const fn shuf8() -> [[u8; 16]; 256] {
    let mut t = [[0u8; 16]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut i) = (0, 0);
        while i < 8 {
            if m >> i & 1 == 1 {
                t[m][k] = i as u8;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_cache_aligned_and_pack_selected_lanes() {
        assert_eq!(&PERM32 as *const _ as usize % 64, 0);
        assert_eq!(&PERM64 as *const _ as usize % 64, 0);
        for m in 0..256usize {
            let want: Vec<u32> = (0..8).filter(|i| m >> i & 1 == 1).collect();
            assert_eq!(&PERM32.0[m][..want.len()], &want[..]);
            let got8: Vec<u32> = SHUF8.0[m][..want.len()].iter().map(|&b| b as u32).collect();
            assert_eq!(got8, want);
            for (k, &i) in want.iter().enumerate() {
                assert_eq!(SHUF16.0[m][2 * k] as u32, 2 * i);
                assert_eq!(SHUF16.0[m][2 * k + 1] as u32, 2 * i + 1);
            }
        }
        for (m, entry) in PERM64.0.iter().enumerate() {
            let want: Vec<u32> = (0..4).filter(|i| m >> i & 1 == 1).collect();
            for (k, &i) in want.iter().enumerate() {
                assert_eq!(entry[2 * k], 2 * i);
                assert_eq!(entry[2 * k + 1], 2 * i + 1);
            }
        }
    }
}
