# rocSHMEM headers

## Copied verbatim from rocSHMEM source

These headers are copied verbatim from the rocSHMEM sources
during RCCL configuration and installation, from CMakeLists.txt.

### Queue Pair core
* `gda/queue_pair/queue_pair_common.hpp`
* `gda/queue_pair/queue_pair_option.hpp`
* `gda/queue_pair/queue_pair_interface.hpp`
* `gda/queue_pair/queue_pair_shmem.hpp`
* `gda/queue_pair/queue_pair_device.hpp`
* `gda/queue_pair_provider.hpp`

### GDA\_MUX (Multiplexing provider)
* `gda/queue_pair_mux.hpp`

### GDA\_IONIC (AMD Pensando IONIC RDMA provider)
* `gda/ionic/queue_pair_ionic.hpp`
* `gda/ionic/provider_gda_ionic.hpp`
* `gda/ionic/ionic_dv.h`
* `gda/ionic/ionic_fw.h`

### GDA\_BNXT (Broadcom BNXT RDMA provider)
* `gda/bnxt/queue_pair_bnxt.hpp`
* `gda/bnxt/provider_gda_bnxt.hpp`
* `gda/bnxt/bnxt_re_dv.h`
* `gda/bnxt/bnxt_re_hsi.h`

### GDA\_MLX5 (Mellanox MLX5 RDMA provider)
* `gda/mlx5/queue_pair_mlx5.hpp`
* `gda/mlx5/provider_gda_mlx5.hpp`
* `gda/mlx5/mlx5dv_core.hpp`
* `gda/mlx5/mlx5_ifc_core.hpp`

### rocSHMEM internals / miscellaneous
* `gda/endian.hpp`
* `gda/gda_enums.hpp`
* `rocshmem/rocshmem_config.h`
* `bit.hpp`
* `constmem.hpp`
* `constants.hpp`

## Replacements provided in RCCL

Replacements for these headers are included directly within the RCCL GIN-GDA codebase.
Some of these already have RCCL analogues (e.g. IBVerbs headers) while for others
only a small portion of the original rocSHMEM header is required.

### RCCL Analogues
* `gda/ibv_core.hpp` (includes RCCL `ibvcore.h` or `<infiniband/verbs.h>`)

### Minimized rocSHMEM header
* `log.hpp`
* `util.hpp`
* `containers/free_list.hpp`
* `rocshmem/rocshmem_common.hpp`
