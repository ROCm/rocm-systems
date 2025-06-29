import sys
import subprocess
import os


def get_gfx_name():
    out = subprocess.check_output("rocminfo | grep 'gfx' | head -n1", shell=True, text=True)
    return out.split("Name:")[-1].strip()

def run(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def main():

    gfx_version = get_gfx_name()

    # First generate the dumps for each testcase along with the debug_trace.txt files
    # This will create the output directory structure for each testcase. 
    # The dumps will be used by the pytest later.
    print(f"Detected gfx version: {gfx_version}")
    for testcase in ("pmc", "sqtt", "spm"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_dir = os.path.join(script_dir, "baselines")
        output_dir = os.path.join(script_dir, "output")
        app = os.path.join(script_dir, "builder_dump")

        out_dir = os.path.join(output_dir, f"{testcase}_builder_dump", gfx_version)
        os.makedirs(out_dir, exist_ok=True)

        debug_trace = os.path.join(out_dir, "debug_trace.txt")
        with open(debug_trace, "w") as f:
            # capture the output of the command to debug_trace.txt
            # it's the same as: $app testcase out_dir gfx_version > out_dir/debug_trace.txt 2>&1
            try:
                subprocess.check_call([app, testcase, out_dir, gfx_version], cwd=script_dir, stdout=f, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Warning: builder_dump failed for {testcase} (exit code {e.returncode}), see {debug_trace} for details.")

    # Now run pytest with the detected gfx_version
    run([sys.executable, "-m", "pytest", "-s", "-v", "--gfx-name", gfx_version], cwd=script_dir)

if __name__ == "__main__":
    main()