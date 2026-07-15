.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-teams:

-------------------------
Team management routines
-------------------------

Predefined teams
----------------

.. cpp:var:: rocshmem_team_t ROCSHMEM_TEAM_WORLD

   Handle that corresponds to the world team that contains all PEs in the
   rocSHMEM program. The default context is tied to the world team.
   Contexts created without specifying a team, default to the world team.
   Available on both host and device.

.. cpp:var:: rocshmem_team_t ROCSHMEM_TEAM_SHARED

   Handle that corresponds to a team of PEs that share a memory domain.
   ``ROCSHMEM_TEAM_SHARED`` refers to the team of all PEs whose symmetric
   heap objects are directly load/store accessible by all PEs in the team
   (i.e. co-located on the same node). Available on both host and device.
   Set to ``ROCSHMEM_TEAM_INVALID`` when IPC is disabled at compile-time
   or runtime, or when the node-local PE ranks are not uniformly strided
   in ``ROCSHMEM_TEAM_WORLD``.

.. cpp:var:: rocshmem_team_t ROCSHMEM_TEAM_INVALID

   A value corresponding to an invalid team. This value can be used to
   initialize or update team handles to indicate that they do not
   reference a valid team. When managed in this way, applications can use
   an equality comparison to test whether a given team handle references a
   valid team. Predefined teams such as ``ROCSHMEM_TEAM_SHARED`` may also
   hold this value when the required capability (e.g. IPC) is unavailable.

ROCSHMEM_TEAM_MY_PE
-------------------

.. cpp:function:: __host__ int rocshmem_team_my_pe(rocshmem_team_t team)
.. cpp:function:: __device__ int rocshmem_team_my_pe(rocshmem_team_t team)

  :param team: The team to query.
  :returns: PE ID of the caller in the provided team.

**Description:**
This routine queries the PE ID of the caller in a team.

ROCSHMEM_TEAM_N_PES
-------------------

.. cpp:function:: __host__ int rocshmem_team_n_pes(rocshmem_team_t team)
.. cpp:function:: __device__ int rocshmem_team_n_pes(rocshmem_team_t team)

  :param team: The team to query.
  :returns: Number of PEs in the provided team.

**Description:**
This routine queries the number of PEs in a team.

ROCSHMEM_TEAM_TRANSLATE_PE
--------------------------

.. cpp:function:: __host__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe, rocshmem_team_t dest_team)

  :param src_team:  Handle of the team from which to translate.
  :param src_pe:    PE-of-interest's index in ``src_team``.
  :param dest_team: Handle of the team to which to translate.
  :returns:         PE of ``src_pe`` in ``dest_team``.
                    If any input is invalid or if ``src_pe`` is
                    not in both source and destination teams, a value of ``-1`` is returned.

**Description:**
This routine translates the PE in ``src_team`` to that in ``dest_team``.

ROCSHMEM_TEAM_SPLIT_STRIDED
---------------------------

.. cpp:function:: __host__ int rocshmem_team_split_strided(rocshmem_team_t parent_team, int start, int stride, int size, const rocshmem_team_config_t *config, long config_mask, rocshmem_team_t *new_team)

  :param parent_team: The team to split from.
  :param start:       The lowest PE number of the subset of the PEs
                      from the parent team that will form the new
                      team.
  :param stride:      The stride between team PE members in the
                      parent team that comprise the subset of PEs
                      that will form the new team.
  :param size:        The number of PEs in the new team.
  :param config:      Pointer to the config parameters for the new team.
  :param config_mask: Bitwise mask representing parameters to use from config.
  :param new_team:    Pointer to the newly created team.
                      If an error occurs during team creation, or if the PE in
                      the parent team is not in the new team, the value will be
                      ``ROCSHMEM_TEAM_INVALID``.

  :returns:           Zero upon successful team creation; non-zero if erroneous.

**Description:**
This routine creates a new a team of PEs. It must be called by all PEs in the parent team.

ROCSHMEM_TEAM_DESTROY
---------------------

.. cpp:function:: __host__ void rocshmem_team_destroy(rocshmem_team_t team)

  :param team: The team to destroy.
               ``ROCSHMEM_TEAM_INVALID``, ``ROCSHMEM_TEAM_WORLD``, and
               ``ROCSHMEM_TEAM_SHARED`` are silently ignored (no-op).
               Passing a handle that was already destroyed or
               never created results in undefined behavior.

  :returns:    None

**Description:**
This routine destroys a team. It must be called by all PEs in the team.
You must destroy all private contexts created in the
team before destroying this team. Otherwise, the behavior
is undefined. This call will destroy only the shareable contexts
created from the referenced team.
