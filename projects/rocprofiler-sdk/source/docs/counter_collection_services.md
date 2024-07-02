# Derived Metrics

## Accumulate metric
### Expression
    expr=accumulate(<basic_level_counter>, <resolution>)
### Description
- The accumulate metric is used to sum the values of a basic level counter over a specified number of cycles. By setting the resolution parameter, you can control the frequency of the summing operation:
    - HIGH_RES: Sums up the basic counter every clock cycle. Captures the value every single cycle for higher accuracy, suitable for fine-grained analysis.
    - LOW_RES: Sums up the basic counter every four clock cycles. Reduces the data points and provides less detailed summing, useful for reducing data volume.
    - NONE: Does nothing and is equivalent to collecting basic_level_counter. Outputs the value of the basic counter without any summing operation.

### Usage (derived_counters.xml)
    <metric name="MeanOccupancyPerCU" expr=accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM descr="Mean occupancy per compute unit."></metric>
- MeanOccupancyPerCU: This metric calculates the mean occupancy per compute unit. It uses the accumulate function with HIGH_RES to sum the SQ_LEVEL_WAVES counter at every clock cycle. This sum is then divided by GRBM_GUI_ACTIVE and the number of compute units (CU_NUM) to derive the mean occupancy.
