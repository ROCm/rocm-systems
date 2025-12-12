#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCm/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI = 
{
    nodeDetails, jobName->

    def prj  = new rocProject('rccl-tests', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = true

    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = [
        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]
    ]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [
        "compute-rocm-dkms-no-npi-hipclang":([ubuntu22:['cpu']])
    ]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 0')])]))
        stage(urlJobName) {
            runCI([ubuntu22:['cpu']], urlJobName)
        }
    }
}
