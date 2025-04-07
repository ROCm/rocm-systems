'
--------------------------------------------------------------------------
Running as root is *strongly* discouraged as any mistake (e.g., in
defining TMPDIR) or bug can result in catastrophic damage to the OS
file system, leaving your system in an unusable state.

We strongly suggest that you run mpirun as a non-root user.

You can override this protection by adding the --allow-run-as-root option
to the cmd line or by setting two environment variables in the following way:
the variable OMPI_ALLOW_RUN_AS_ROOT=1 to indicate the desire to override this
protection, and OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 to confirm the choice and
add one more layer of certainty that you want to do so.
We reiterate our advice against doing so - please proceed at your own risk.
--------------------------------------------------------------------------
'

MPIRUN=mpirun
if ! command -v $MPIRUN &> /dev/null
then
    echo "$MPIRUN could not be found. checking libs"
    if [ -f "/usr/lib64/openmpi/bin/mpirun" ]
    then
        MPIRUN=/usr/lib64/openmpi/bin/mpirun
    else
        if [ -f "/usr/lib64/mpi/gcc/openmpi2/bin/mpirun" ]
        then
            MPIRUN=/usr/lib64/mpi/gcc/openmpi2/bin/mpirun
        else
            echo "$MPIRUN could not be found. exiting"
            exit
        fi
    fi
fi
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
$MPIRUN --allow-run-as-root -np 2 $SCRIPTPATH/mpi_vectoradd mdrun -pin on -nsteps 10 -resetstep 9 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0
