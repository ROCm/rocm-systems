//! ROCm GPU visibility filtering, normalization, and DBT host selection.
//!
//! DBT guest mode is the one place where two GPUs have to coexist in a
//! single process: the synthetic guest, so ROCR can build the agent the
//! application queries for its guest identity, and the real host, so
//! queues, allocations and code loading keep working. Everything in this
//! module exists to keep both internally visible while the HSA hook
//! shadows the guest into the host's public agent slot.
//!
//! Two runtimes read the visibility environment, and they do not agree,
//! so the two filters here are deliberately different:
//!
//! * `ROCR_VISIBLE_DEVICES` is applied first, by ROCR, over the KFD
//!   topology. It uppercases its tokens, treats a repeat or an
//!   unparseable token as the end of the selection, and stops on an
//!   ambiguous UUID prefix.
//! * `HIP_VISIBLE_DEVICES` (or `CUDA_VISIBLE_DEVICES` as its fallback)
//!   is applied afterwards, by CLR, over what ROCR left. It is
//!   case-sensitive, takes the *first* UUID match rather than
//!   terminating on ambiguity, and skips a repeat without ending the
//!   selection.
//!
//! Those asymmetries are not tidiness the port can smooth over: an
//! application's device order is what they produce, and getting one
//! wrong silently runs the workload on the wrong GPU.

/// GPU identity used while applying ROCm visibility selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibleGpu {
    /// Ordinal in the current KFD enumeration.
    pub ordinal: u32,
    /// Nonzero KFD topology GPU ID.
    pub gpu_id: u32,
    /// Numeric GFX target version.
    pub gfx_target_version: u32,
    /// KFD unique ID used in GPU UUID selectors.
    pub unique_id: u64,
}

/// Outcome of picking a DBT host GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSelection {
    /// A visible GPU satisfies the request; carries its GPU ID.
    Selected(u32),
    /// The configured GPU is not client-visible.
    ExplicitGpuHidden(u32),
    /// The configured GPU does not match the host ISA.
    ExplicitGpuIsaMismatch(u32),
    /// No client-visible GPU matches the host ISA.
    NoIsaMatch,
}

/// Environment variable replacement produced by visibility normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibilityOverride {
    /// The variable to set: `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`.
    pub name: String,
    /// Its new value: a canonical comma-separated list of ordinals.
    pub value: String,
}

/// The `GPU-` prefix every UUID selector carries.
const GPU_UUID_PREFIX: &str = "GPU-";

/// What ROCR reports for an agent with no UUID support, so it never
/// names a device.
const NO_UUID_SENTINEL: &str = "GPU-XX";

/// `GPU-` plus at least one hex digit.
const MIN_UUID_SELECTOR_LEN: usize = GPU_UUID_PREFIX.len() + 1;

/// `GPU-` plus all 16 hex digits of the 64-bit unique ID.
const MAX_UUID_SELECTOR_LEN: usize = GPU_UUID_PREFIX.len() + (u64::BITS as usize / 4);

/// The canonical UUID spelling for a KFD unique ID.
fn gpu_uuid(unique_id: u64) -> String {
    format!("GPU-{unique_id:016X}")
}

/// Whether `token` is spelled as a GPU UUID prefix rather than a device
/// ordinal.
///
/// The accepted window is ROCR's own: `GPU-` followed by between one and
/// sixteen hex digits of the unique ID. Both selector paths share this
/// predicate; only the matching below each call site differs.
fn is_gpu_uuid_selector(token: &str) -> bool {
    token.len() >= MIN_UUID_SELECTOR_LEN
        && token.len() <= MAX_UUID_SELECTOR_LEN
        && token.starts_with(GPU_UUID_PREFIX)
        && token != NO_UUID_SENTINEL
}

/// Parse a token exactly as `strtol(token, &end, 0)` followed by a
/// `*end == '\0'` check would: base is inferred from the prefix (`0x`
/// hexadecimal, leading `0` octal, otherwise decimal), and any trailing
/// character rejects the whole token.
///
/// Base 0 rather than base 10 is what ROCR does, and it is load-bearing
/// in one direction only: it makes `010` mean 8. Partial consumption is
/// rejected either way, so every token this returns `None` for is one
/// the C++ also refused.
fn parse_selector_base0(token: &str) -> Option<i64> {
    let (negative, rest) = match token.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, token.strip_prefix('+').unwrap_or(token)),
    };
    let (radix, digits) =
        if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            (16, hex)
        } else if rest.len() > 1 && rest.starts_with('0') {
            (8, &rest[1..])
        } else {
            (10, rest)
        };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    let value = i64::from_str_radix(digits, radix).ok()?;
    Some(if negative { -value } else { value })
}

/// Remove non-GPU KFD nodes and assign compact GPU ordinals.
///
/// A `gpu_id` of zero is a CPU node. Dropping those and renumbering is
/// what makes an ordinal in a visibility selector mean the same thing
/// here as it does inside ROCR.
#[must_use]
pub fn enumerate_kfd_gpus(candidates: &[VisibleGpu]) -> Vec<VisibleGpu> {
    let mut gpus = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.gpu_id == 0 {
            continue;
        }
        let mut gpu = *candidate;
        gpu.ordinal = u32::try_from(gpus.len()).unwrap_or(u32::MAX);
        gpus.push(gpu);
    }
    gpus
}

/// Apply `ROCR_VISIBLE_DEVICES` filtering and ordering semantics.
///
/// `None` means the variable is unset and every GPU stays visible; an
/// empty string means it is set to nothing, which hides all of them.
/// That difference is the whole reason this takes an `Option` rather
/// than a `&str`.
#[must_use]
pub fn filter_rocr_visible_gpus(gpus: &[VisibleGpu], selector: Option<&str>) -> Vec<VisibleGpu> {
    let Some(selector) = selector else {
        return gpus.to_vec();
    };
    let mut filtered: Vec<VisibleGpu> = Vec::new();
    for raw in selector.split(',') {
        if filtered.len() >= gpus.len() {
            break;
        }
        let token = raw.trim().to_uppercase();

        let mut index: Option<usize> = None;
        if is_gpu_uuid_selector(&token) {
            for (candidate, gpu) in gpus.iter().enumerate() {
                if gpu.unique_id == 0 || !gpu_uuid(gpu.unique_id).starts_with(&token) {
                    continue;
                }
                if index.is_some() {
                    // Ambiguous prefix: ROCR stops here rather than
                    // guessing, and so must we.
                    return filtered;
                }
                index = Some(candidate);
            }
        } else if let Some(parsed) = parse_selector_base0(&token)
            && parsed >= 0
        {
            index = usize::try_from(parsed).ok();
        }

        let Some(index) = index.filter(|i| *i < gpus.len()) else {
            return filtered;
        };
        if filtered
            .iter()
            .any(|gpu| gpu.ordinal == gpus[index].ordinal)
        {
            return filtered;
        }
        filtered.push(gpus[index]);
    }
    filtered
}

/// Apply `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` filtering
/// semantics.
///
/// Three deliberate departures from [`filter_rocr_visible_gpus`], each
/// matching CLR rather than ROCR: the comparison is case-sensitive
/// because tokens are never uppercased, the first UUID match wins
/// instead of ambiguity ending the selection, and a repeated ordinal is
/// skipped without ending it.
///
/// The UUID spelling guard is stricter than CLR's own, which gates on a
/// bare substring search and would resolve the token `GPU-` to agent
/// zero. Diverging is safe only because CLR never sees a token this
/// rejected: in guest mode the client selector is always rewritten to
/// [`normalized_client_visible_devices`]'s canonical numeric list before
/// the workload is executed.
#[must_use]
pub fn filter_client_visible_gpus(gpus: &[VisibleGpu], selector: Option<&str>) -> Vec<VisibleGpu> {
    let Some(selector) = selector else {
        return gpus.to_vec();
    };
    let mut filtered: Vec<VisibleGpu> = Vec::new();
    for raw in selector.split(',') {
        if filtered.len() >= gpus.len() {
            break;
        }
        let mut token = raw.to_string();
        if is_gpu_uuid_selector(&token) {
            for (candidate, gpu) in gpus.iter().enumerate() {
                if gpu.unique_id != 0 && gpu_uuid(gpu.unique_id).starts_with(&token) {
                    token = candidate.to_string();
                    break;
                }
            }
        }

        // Strict decimal: `from_chars` accepts no sign and no leading
        // zero, and round-tripping the parse is how the C++ enforces
        // that. `01` and `00` are rejected, not read as 1 and 0.
        let Ok(index) = token.parse::<u32>() else {
            return filtered;
        };
        let index = index as usize;
        if token != index.to_string() || index >= gpus.len() {
            return filtered;
        }
        if !filtered.iter().any(|gpu| gpu.gpu_id == gpus[index].gpu_id) {
            filtered.push(gpus[index]);
        }
    }
    filtered
}

/// The client selector in effect: `HIP_VISIBLE_DEVICES` when set and
/// non-empty, otherwise `CUDA_VISIBLE_DEVICES` on the same terms.
fn client_selector<'a>(
    hip_visible: Option<&'a str>,
    cuda_visible: Option<&'a str>,
) -> Option<(&'static str, &'a str)> {
    if let Some(hip) = hip_visible.filter(|v| !v.is_empty()) {
        return Some(("HIP_VISIBLE_DEVICES", hip));
    }
    cuda_visible
        .filter(|v| !v.is_empty())
        .map(|cuda| ("CUDA_VISIBLE_DEVICES", cuda))
}

/// GPUs visible after applying the ROCR and client selectors in the
/// order the runtimes apply them.
#[must_use]
pub fn effective_visible_gpus(
    topology: &[VisibleGpu],
    rocr_visible: Option<&str>,
    hip_visible: Option<&str>,
    cuda_visible: Option<&str>,
) -> Vec<VisibleGpu> {
    let gpus = filter_rocr_visible_gpus(topology, rocr_visible);
    match client_selector(hip_visible, cuda_visible) {
        Some((_, selector)) => filter_client_visible_gpus(&gpus, Some(selector)),
        None => gpus,
    }
}

/// Normalize or synthesize a client selector using post-ROCR numeric
/// ordinals.
///
/// CLR applies its selector *after* HSA agent iteration, by which point
/// the hook has already replaced the host's public identity with the
/// guest's. A UUID naming the host would then resolve against an agent
/// that no longer reports that UUID, so the selector is rewritten to
/// ordinals while the mapping is still knowable.
///
/// `first_gpu_id` is the host chosen by [`select_host_gpu`] over the
/// same selectors; when given, it is rotated to the front so the
/// application's device zero is the GPU the guest actually runs on.
/// Without an existing client selector a `HIP_VISIBLE_DEVICES` is
/// synthesized only when a host was selected, since HIP is the primary
/// ROCm client visibility control.
#[must_use]
pub fn normalized_client_visible_devices(
    topology: &[VisibleGpu],
    rocr_visible: Option<&str>,
    hip_visible: Option<&str>,
    cuda_visible: Option<&str>,
    first_gpu_id: Option<u32>,
) -> Option<VisibilityOverride> {
    let client = client_selector(hip_visible, cuda_visible);
    if client.is_none() && first_gpu_id.is_none() {
        return None;
    }

    let rocr_gpus = filter_rocr_visible_gpus(topology, rocr_visible);
    let mut selected = match client {
        Some((_, selector)) => filter_client_visible_gpus(&rocr_gpus, Some(selector)),
        None => rocr_gpus.clone(),
    };
    if let Some(first_gpu_id) = first_gpu_id
        && let Some(at) = selected.iter().position(|gpu| gpu.gpu_id == first_gpu_id)
    {
        selected[..=at].rotate_right(1);
    }

    let ordinals: Vec<String> = selected
        .iter()
        .filter_map(|gpu| {
            rocr_gpus
                .iter()
                .position(|candidate| candidate.gpu_id == gpu.gpu_id)
        })
        .map(|index| index.to_string())
        .collect();
    let name = client.map_or("HIP_VISIBLE_DEVICES", |(name, _)| name);
    Some(VisibilityOverride {
        name: name.to_string(),
        value: ordinals.join(","),
    })
}

/// Append the synthetic DBT guest ordinal to a non-empty ROCR selection.
///
/// The guest node is appended to the topology, so its ordinal is the
/// old length. Adding it keeps both the guest and the selected host
/// internally visible to ROCR, which is the precondition for the hook's
/// public shadowing. Returns `None` when the rewrite would be a no-op.
#[must_use]
pub fn expanded_rocr_visible_devices(
    topology: &[VisibleGpu],
    rocr_visible: Option<&str>,
) -> Option<String> {
    let rocr_visible = rocr_visible?;
    if topology.is_empty() {
        return None;
    }
    let selected = filter_rocr_visible_gpus(topology, Some(rocr_visible));
    if selected.is_empty() {
        return None;
    }

    let mut ordinals: Vec<String> = selected.iter().map(|gpu| gpu.ordinal.to_string()).collect();
    ordinals.push(topology.len().to_string());
    let rewritten = ordinals.join(",");
    (rewritten != rocr_visible).then_some(rewritten)
}

/// Select an explicit or automatic DBT host from the client-visible
/// GPUs.
///
/// A configured `gpu_id` of zero means "first ISA match"; anything else
/// pins a GPU, and the pin is refused if standard ROCm visibility
/// settings hide it or it is the wrong architecture. Refusing is the
/// point: silently falling back would run the workload somewhere the
/// user did not ask for.
#[must_use]
pub fn select_host_gpu(
    visible_gpus: &[VisibleGpu],
    configured_gpu_id: u32,
    gfx_target_version: u32,
) -> HostSelection {
    if configured_gpu_id != 0 {
        let Some(gpu) = visible_gpus
            .iter()
            .find(|gpu| gpu.gpu_id == configured_gpu_id)
        else {
            return HostSelection::ExplicitGpuHidden(configured_gpu_id);
        };
        return if gpu.gfx_target_version == gfx_target_version {
            HostSelection::Selected(configured_gpu_id)
        } else {
            HostSelection::ExplicitGpuIsaMismatch(configured_gpu_id)
        };
    }

    visible_gpus
        .iter()
        .find(|gpu| gpu.gfx_target_version == gfx_target_version)
        .map_or(HostSelection::NoIsaMatch, |gpu| {
            HostSelection::Selected(gpu.gpu_id)
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn gpu(ordinal: u32, gpu_id: u32, gfx_target_version: u32, unique_id: u64) -> VisibleGpu {
        VisibleGpu {
            ordinal,
            gpu_id,
            gfx_target_version,
            unique_id,
        }
    }

    fn test_gpus() -> Vec<VisibleGpu> {
        vec![
            gpu(0, 100, 90402, 0x1111_1111_1111_1111),
            gpu(1, 101, 90402, 0x2222_2222_2222_2222),
            gpu(2, 102, 120001, 0x3333_3333_3333_3333),
        ]
    }

    fn gpu_ids(gpus: &[VisibleGpu]) -> Vec<u32> {
        gpus.iter().map(|gpu| gpu.gpu_id).collect()
    }

    #[test]
    fn unset_and_empty_rocr_selectors_differ() {
        let gpus = test_gpus();
        assert_eq!(filter_rocr_visible_gpus(&gpus, None).len(), 3);
        assert!(filter_rocr_visible_gpus(&gpus, Some("")).is_empty());
    }

    #[test]
    fn kfd_enumeration_skips_zero_gpu_ids_and_compacts_ordinals() {
        let candidates = vec![
            gpu(0, 0, 90402, 0),
            gpu(1, 101, 90402, 0x2222_2222_2222_2222),
        ];
        let gpus = enumerate_kfd_gpus(&candidates);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].ordinal, 0);
        assert_eq!(gpus[0].gpu_id, 101);
    }

    #[test]
    fn numeric_and_uuid_selectors_reorder_devices() {
        let gpus = test_gpus();
        let numeric = filter_rocr_visible_gpus(&gpus, Some("1,0"));
        let uuid = filter_rocr_visible_gpus(&gpus, Some("gpu-2222"));

        assert_eq!(gpu_ids(&numeric), vec![101, 100]);
        assert_eq!(gpu_ids(&uuid), vec![101]);
    }

    #[test]
    fn ambiguous_rocr_uuid_terminates_after_valid_prefix() {
        let gpus = vec![
            gpu(0, 100, 90402, 0x1111_1111_1111_1111),
            gpu(1, 101, 90402, 0x2222_2222_1111_1111),
            gpu(2, 102, 90402, 0x2222_2222_4444_4444),
        ];
        let selected = filter_rocr_visible_gpus(&gpus, Some("0,GPU-2222,1"));

        assert_eq!(gpu_ids(&selected), vec![100]);
    }

    #[test]
    fn client_uuid_uses_case_sensitive_first_match() {
        let gpus = vec![
            gpu(0, 100, 90402, 0x1111_1111_1111_1111),
            gpu(1, 101, 90402, 0x1111_1111_1111_1122),
        ];
        let first = filter_client_visible_gpus(&gpus, Some("GPU-1111"));
        let lowercase = filter_client_visible_gpus(&gpus, Some("gpu-1111"));

        assert_eq!(gpu_ids(&first), vec![100]);
        assert!(lowercase.is_empty());
    }

    #[test]
    fn invalid_rocr_token_preserves_valid_prefix() {
        let selected = filter_rocr_visible_gpus(&test_gpus(), Some("1,invalid,0"));
        assert_eq!(gpu_ids(&selected), vec![101]);
    }

    #[test]
    fn malformed_selectors_terminate_selection() {
        assert!(filter_rocr_visible_gpus(&test_gpus(), Some("GPU-")).is_empty());
        assert!(filter_client_visible_gpus(&test_gpus(), Some("01")).is_empty());
        assert!(filter_client_visible_gpus(&test_gpus(), Some("00")).is_empty());

        // The client UUID guard rejects malformed spellings rather than
        // substring-matching them onto the first agent: too short to
        // carry a body, the ROCR no-UUID sentinel, and one hex digit
        // past the 20-character maximum. Each falls through to the
        // numeric parse, which terminates selection.
        assert!(filter_client_visible_gpus(&test_gpus(), Some("GPU-")).is_empty());
        assert!(filter_client_visible_gpus(&test_gpus(), Some("GPU-XX")).is_empty());
        assert!(filter_client_visible_gpus(&test_gpus(), Some("GPU-11111111111111111")).is_empty());

        let rocr_prefix = filter_rocr_visible_gpus(&test_gpus(), Some("0,GPU-,1"));
        assert_eq!(gpu_ids(&rocr_prefix), vec![100]);

        let client_prefix = filter_client_visible_gpus(&test_gpus(), Some("1,00,2"));
        assert_eq!(gpu_ids(&client_prefix), vec![101]);

        let client_uuid_prefix = filter_client_visible_gpus(&test_gpus(), Some("1,GPU-,2"));
        assert_eq!(gpu_ids(&client_uuid_prefix), vec![101]);
    }

    #[test]
    fn duplicate_and_reordered_selectors_match_runtime_behavior() {
        let rocr_duplicate = filter_rocr_visible_gpus(&test_gpus(), Some("0,0,1"));
        assert_eq!(gpu_ids(&rocr_duplicate), vec![100]);

        let client_duplicate = filter_client_visible_gpus(&test_gpus(), Some("0,0,1"));
        assert_eq!(gpu_ids(&client_duplicate), vec![100, 101]);

        let client_reordered = filter_client_visible_gpus(&test_gpus(), Some("2,0"));
        assert_eq!(gpu_ids(&client_reordered), vec![102, 100]);
    }

    #[test]
    fn negative_rocr_ordinal_preserves_valid_prefix() {
        let selected = filter_rocr_visible_gpus(&test_gpus(), Some("0,-1,1"));
        assert_eq!(gpu_ids(&selected), vec![100]);
    }

    #[test]
    fn hip_selector_uses_post_rocr_ordinals() {
        let selected = effective_visible_gpus(&test_gpus(), Some("1,0"), Some("1"), None);
        assert_eq!(gpu_ids(&selected), vec![100]);
    }

    #[test]
    fn hip_selector_takes_precedence_over_cuda_fallback() {
        let hip = effective_visible_gpus(&test_gpus(), None, Some("1"), Some("2"));
        let cuda = effective_visible_gpus(&test_gpus(), None, Some(""), Some("2"));

        assert_eq!(gpu_ids(&hip), vec![101]);
        assert_eq!(gpu_ids(&cuda), vec![102]);
    }

    #[test]
    fn normalizes_client_uuid_to_post_rocr_ordinal() {
        let normalized = normalized_client_visible_devices(
            &test_gpus(),
            Some("1,0"),
            Some("GPU-1111111111111111"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(normalized.name, "HIP_VISIBLE_DEVICES");
        assert_eq!(normalized.value, "1");
    }

    #[test]
    fn normalizes_client_selector_against_expanded_rocr_order() {
        let normalized =
            normalized_client_visible_devices(&test_gpus(), Some("1,0,3"), Some("1"), None, None)
                .unwrap();
        assert_eq!(normalized.name, "HIP_VISIBLE_DEVICES");
        assert_eq!(normalized.value, "1");
    }

    #[test]
    fn normalizes_selected_dbt_host_to_client_device_zero() {
        let heterogeneous = vec![
            gpu(0, 100, 110000, 0x1111_1111_1111_1111),
            gpu(1, 101, 90402, 0x2222_2222_2222_2222),
        ];
        let normalized =
            normalized_client_visible_devices(&heterogeneous, None, Some("0,1"), None, Some(101))
                .unwrap();

        assert_eq!(normalized.name, "HIP_VISIBLE_DEVICES");
        assert_eq!(normalized.value, "1,0");
    }

    #[test]
    fn synthesizes_client_order_for_selected_dbt_host() {
        let heterogeneous = vec![
            gpu(0, 100, 110000, 0x1111_1111_1111_1111),
            gpu(1, 101, 90402, 0x2222_2222_2222_2222),
        ];
        let normalized =
            normalized_client_visible_devices(&heterogeneous, None, None, None, Some(101)).unwrap();

        assert_eq!(normalized.name, "HIP_VISIBLE_DEVICES");
        assert_eq!(normalized.value, "1,0");
    }

    #[test]
    fn does_not_synthesize_client_order_without_selected_host() {
        assert!(normalized_client_visible_devices(&test_gpus(), None, None, None, None).is_none());
    }

    #[test]
    fn selects_dbt_host_from_client_visible_gpus_before_normalization() {
        let visible = effective_visible_gpus(&test_gpus(), None, Some("1"), None);
        let automatic = select_host_gpu(&visible, 0, 90402);
        assert_eq!(automatic, HostSelection::Selected(101));

        let normalized =
            normalized_client_visible_devices(&test_gpus(), None, Some("1"), None, Some(101))
                .unwrap();
        assert_eq!(normalized.value, "1");

        assert_eq!(
            select_host_gpu(&visible, 100, 90402),
            HostSelection::ExplicitGpuHidden(100)
        );
    }

    #[test]
    fn expands_rocr_selection_with_guest_ordinal() {
        let expanded = expanded_rocr_visible_devices(&test_gpus(), Some("GPU-2222")).unwrap();
        assert_eq!(expanded, "1,3");
    }

    #[test]
    fn does_not_expand_empty_rocr_selection() {
        assert!(expanded_rocr_visible_devices(&test_gpus(), Some("5")).is_none());
    }

    #[test]
    fn selects_first_visible_isa_match() {
        assert_eq!(
            select_host_gpu(&test_gpus(), 0, 90402),
            HostSelection::Selected(100)
        );
    }

    #[test]
    fn selects_explicit_gpu_with_matching_isa() {
        assert_eq!(
            select_host_gpu(&test_gpus(), 101, 90402),
            HostSelection::Selected(101)
        );
    }

    #[test]
    fn rejects_hidden_explicit_gpu() {
        assert_eq!(
            select_host_gpu(&test_gpus()[..1], 101, 90402),
            HostSelection::ExplicitGpuHidden(101)
        );
    }

    #[test]
    fn rejects_explicit_gpu_with_different_isa() {
        assert_eq!(
            select_host_gpu(&test_gpus(), 102, 90402),
            HostSelection::ExplicitGpuIsaMismatch(102)
        );
    }

    #[test]
    fn reports_missing_isa_match() {
        assert_eq!(
            select_host_gpu(&test_gpus(), 0, 999_999),
            HostSelection::NoIsaMatch
        );
    }

    #[test]
    fn base0_parsing_matches_strtol() {
        assert_eq!(parse_selector_base0("0"), Some(0));
        assert_eq!(parse_selector_base0("010"), Some(8));
        assert_eq!(parse_selector_base0("0x10"), Some(16));
        assert_eq!(parse_selector_base0("-1"), Some(-1));
        assert_eq!(parse_selector_base0("08"), None);
        assert_eq!(parse_selector_base0("0x"), None);
        assert_eq!(parse_selector_base0(""), None);
        assert_eq!(parse_selector_base0("invalid"), None);
    }
}
