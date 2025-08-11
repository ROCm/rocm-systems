# Runtime API Registration

## Services

- HIP runtime table registration

## Properties

- `api_registration_callback` function validates the type of library being intercepted, ensures there is only one instance of the HIP runtime library, and retrieves the dispatch table containing the API functions.
- Collects a "call stack" of intercepted API calls.
