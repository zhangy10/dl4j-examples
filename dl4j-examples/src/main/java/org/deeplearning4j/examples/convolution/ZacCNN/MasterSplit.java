package org.deeplearning4j.examples.convolution.ZacCNN;

import java.util.ArrayList;
import java.util.List;

public class MasterSplit {


    public static void main(String[] args) {
        DataType type = DataType.TEST;
        int task = 1;

        // split task
        List<TrainSplit.Pair> list = TrainSplit.getTask(task, DataSet.getConfig(type).getTaskNum());
        System.out.println(list.toString());
        // assgin task
        TrainSplit master = null;
        List<TrainSplit> slaveList = new ArrayList<>();
        for (int i = 0; i < task; i++) {
            TrainSplit.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            if (i == 0) {
                master = new TrainSplit(i, config.setTaskRange(fragment.start, fragment.end), task - 1);
            } else {
                TrainSplit slave = new TrainSplit(master.getQueue(), i, config.setTaskRange(fragment.start, fragment.end));
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }
        master.start();
        for (TrainSplit t : slaveList) {
            t.start();
        }
    }
}
