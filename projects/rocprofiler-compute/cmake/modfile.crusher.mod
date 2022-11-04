-- Crusher-specific additions
depends_on "cray-python"
depends_on "rocm"
prereq(atleast("rocm","5.1.0"))
