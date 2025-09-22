#include "common_utility.h"


double CalcMedian(vector<double> scores)
{
	double median;
	size_t size = scores.size();

	if (size  % 2 == 0)
		median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
	else 
		median = scores[size / 2];

	return median;
}

double CalcMean(vector<double> scores)
{
	double mean = 0;
	size_t size = scores.size();

       for (int i=0; i<size; ++i)
	   	mean += scores[i];

	return mean/size;
}


double CalcStdDeviation(vector<double> scores, int score_mean)
{
	double ret = 0.0;
	for (int i=0; i<scores.size(); ++i)
	{
		ret += (scores[i] - score_mean) * (scores[i] - score_mean);
	}

	ret /= scores.size();

	return sqrt(ret);
}

int CalcConcurrentQueues(vector<double> scores)
{
    int num_of_concurrent_queues = 0;
    vector<double>execpted_exec_time_array;
    
    for (int i=0; i<scores.size(); ++i)
    {
        execpted_exec_time_array.push_back(scores[0]/(1<<i));
    }

   
   for (int i=0; i<scores.size(); ++i)
   {
	   cout << "expected exe time = " << execpted_exec_time_array[i] << endl;
   }

    for (int i=1; i<scores.size(); ++i)
    {
        if ((execpted_exec_time_array[i] - scores[i]) < 0.1 * execpted_exec_time_array[i])
            ++num_of_concurrent_queues;
    }

    return num_of_concurrent_queues;
}


