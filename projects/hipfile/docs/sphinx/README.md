We do not check in requirements.txt because pinned transitive
dependencies frequently trigger dependency vulnerability alerts.

To install the requirements, just run `pip install -r requirements.in`
and that will install everything needed for the docs without having
to generate requirements.txt.
