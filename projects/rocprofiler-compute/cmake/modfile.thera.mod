-- Thera-specific additions
depends_on "python"
depends_on "rocm"
prereq(atleast("rocm","5.2.0"))
local home = os.getenv("HOME")
setenv("MPLCONFIGDIR",pathJoin(home,".matplotlib"))
