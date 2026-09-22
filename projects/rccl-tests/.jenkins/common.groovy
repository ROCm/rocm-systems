// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String hipclangArgs = jobName.contains('hipclang') ? '--hip-clang' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.build_prefix}
                git clone --recursive https://github.com/ROCm/rccl.git
                cd rccl
                ./install.sh -l
                cd ../..
                ${auxiliary.exitIfNotSuccess()}
                
                cd ${project.paths.project_build_prefix}
                export RCCL_DIR=\$(pwd)/../rccl/build/release
                ./install.sh --rccl_home \$RCCL_DIR
                ${auxiliary.exitIfNotSuccess()}
            """

    platform.runCommand(this,command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                python3 -m pip install --upgrade pytest
                python3 -m pytest --version
                python3 -m pytest -k "not MPI and not host and not fine" --verbose --junitxml=./testreport.xml
            """

   platform.runCommand(this, command)
}

return this
