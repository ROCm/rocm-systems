.. meta::
  :description: The ROCm Data Center tool (RDC) addresses key infrastructure challenges regarding AMD GPUs in cluster and data center environments and simplifies their administration
  :keywords: Job stats use case, RDC feature example, ROCm Data Center feature sample, RDC feature sample, ROCm Data Center feature example

.. _job-stats-sample:

**********************
Job stats sample code
**********************

The following pseudocode shows how RDC API can be directly used to record GPU statistics associated with any job or workload. Refer to the `example code <https://github.com/ROCm/rdc/tree/amd-staging/example>`_ on how to build it.

For more information on Job stats, see :ref:`Job stats <job-stats>`.

.. code-block:: shell

    //Initialize the RDC
    rdc_handle_t rdc_handle;
    rdc_status_t result=rdc_init(0);

    //Dynamically choose to run in standalone or embedded mode
    bool standalone = false;
    std::cin>> standalone;
    if (standalone)
      result = rdc_connect("127.0.0.1:50051", &rdc_handle, nullptr, nullptr, nullptr); //It will connect to the daemon
    else
      result = rdc_start_embedded(RDC_OPERATION_MODE_MANUAL, &rdc_handle); //call library directly, here we run embedded in manual mode

    //Now we can use the same API for both standalone and embedded
    //(1) create group
    rdc_gpu_group_t groupId;
    result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "MyGroup1", &groupId);

    //(2) Add the GPUs to the group
    result = rdc_group_gpu_add(rdc_handle, groupId, 0); //Add GPU 0
    result = rdc_group_gpu_add(rdc_handle, groupId, 1); //Add GPU 1

    //(3) start the recording the Slurm job 123. Set the sample frequency to once per second
    result = rdc_job_start_stats(rdc_handle, group_id,
      "123", 1000000);

    //For standalone mode, the daemon will update and cache the samples
    //In manual mode, we must call rdc_field_update_all periodically to take samples
    if (!standalone) { //embedded manual mode
      for (int i=5; i>0; i--) { //As an example, we will take 5 samples
      result = rdc_field_update_all(rdc_handle, 0);
      usleep(1000000);
      }
    } else { //standalone mode, do nothing
      usleep(5000000); //sleep 5 seconds before fetch the stats
    }

    //(4) stop the Slurm job 123, which will stop the watch
    // Note: we do not have to stop the job to get stats. The rdc_job_get_stats can be called at any time before stop
    result = rdc_job_stop_stats(rdc_handle, "123");

    //(5) Get the stats
    rdc_job_info_t job_info;
    result = rdc_job_get_stats(rdc_handle, "123", &job_info);
    std::cout<<"Average Memory Utilization: " <<job_info.summary.memoryUtilization.average <<std::endl;

    //The cleanup and shutdown ....
