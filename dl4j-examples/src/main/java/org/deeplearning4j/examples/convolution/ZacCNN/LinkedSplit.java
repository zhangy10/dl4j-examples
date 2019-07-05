package org.deeplearning4j.examples.convolution.ZacCNN;


import java.util.ArrayList;
import java.util.List;

/**
 * 1 to 1 sending
 */
public class LinkedSplit {


    public static void main(String[] args) {
        DataType type = DataType.EMG;
        int taskN = 3;

        List<TrainSplit.Pair> list = TrainSplit.getTask(taskN, DataSet.getConfig(type).getTaskNum());

        System.out.println(list.toString());

        List<TrainSplit> taskList = new ArrayList<>();
        TrainSplit root = null;
        TrainSplit master = null;
        for (int i = 0; i < taskN; i++) {
            // task
            TrainSplit.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            TrainSplit task = null;
            if (root == null) {
                // master 1st
                task = new TrainSplit(i, config.setTaskRange(fragment.start, fragment.end), taskN - 1, true);
                master = task;
            } else {
                task = new TrainSplit(root.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), true);
                // last one will not wait for any nodes
                if (i == taskN - 1) {
                    task.isEnd = true;
                }
                // for broadcast
                master.addSlave(task.getQueue());
            }
            root = task;
            taskList.add(task);
        }

        for (TrainSplit t : taskList) {
            t.start();
        }
    }
}
