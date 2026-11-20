// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process CU restrictions applied after visible-agent ordering.

/// Parses decimal IDs and inclusive ranges, bounding expansion by topology.
fn elements(input: &str, maximum: u32) -> Option<Vec<u32>> {
    let mut values = Vec::new();
    for part in input.split(',') {
        let mut range = part.split('-');
        let first = range.next()?;
        if first.is_empty() || !first.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let first = first.parse::<u32>().ok()?;
        let last = if let Some(last) = range.next() {
            if last.is_empty() || !last.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            last.parse::<u32>().ok()?
        } else {
            first
        };
        if range.next().is_some() || last < first {
            return None;
        }
        values.extend(first..=last.min(maximum));
    }
    values.sort_unstable();
    if values.windows(2).any(|pair| pair[0] == pair[1]) || values.is_empty() {
        return None;
    }
    Some(values)
}

/// Parses `ROCr`'s GPU-list:CU-list sets, retaining the valid prefix on error.
pub(crate) fn parse(input: &str, compute_units: &[u32]) -> Vec<Option<Vec<u32>>> {
    let mut masks = vec![None; compute_units.len()];
    let Some(maximum_gpu) = u32::try_from(compute_units.len())
        .ok()
        .and_then(|count| count.checked_sub(1))
    else {
        return masks;
    };
    let maximum_cu = compute_units
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_sub(1);
    let input = input
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();
    for set in input.split(';') {
        let Some((gpus, cus)) = set.split_once(':') else {
            break;
        };
        let Some(gpus) = elements(gpus, maximum_gpu) else {
            break;
        };
        if gpus.iter().any(|gpu| masks[*gpu as usize].is_some()) {
            break;
        }
        let mask = if let Some(hex) = cus.strip_prefix("0X") {
            if hex.is_empty() || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                break;
            }
            let mut mask = Vec::new();
            for chunk in hex
                .as_bytes()
                .rchunks(8)
                .take((maximum_cu / 32 + 1) as usize)
            {
                let Ok(text) = std::str::from_utf8(chunk) else {
                    break;
                };
                let Ok(word) = u32::from_str_radix(text, 16) else {
                    break;
                };
                mask.push(word);
            }
            mask
        } else {
            let Some(cus) = elements(cus, maximum_cu) else {
                break;
            };
            let mut mask = vec![0; (maximum_cu / 32 + 1) as usize];
            for cu in cus {
                mask[(cu / 32) as usize] |= 1 << (cu % 32);
            }
            mask
        };
        if mask.iter().all(|word| *word == 0) {
            break;
        }
        for gpu in gpus {
            masks[gpu as usize] = Some(mask.clone());
        }
    }
    masks
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn lists_ranges_and_hex_masks_follow_visible_agent_order() {
        assert_eq!(
            parse(" 1-2:0-3,8-9; 0:0x30000000f ", &[64, 32, 64]),
            [
                Some(vec![15, 3]),
                Some(vec![0x30f, 0]),
                Some(vec![0x30f, 0])
            ]
        );
        assert_eq!(parse("0:32-1000000000", &[64]), [Some(vec![0, u32::MAX])]);
    }

    #[test]
    fn syntax_errors_stop_before_publishing_the_bad_set() {
        for suffix in ["1:0x0", "1:1-0", "1:0,0", "1:bogus", "0:0-1", "1::1"] {
            let input = format!("0:0-3;{suffix};2:0-1");
            assert_eq!(parse(&input, &[32, 32, 32]), [Some(vec![15]), None, None]);
        }
    }
}
