package org.deeplearning4j.examples.convolution.ZacCNN;


import java.util.ArrayList;
import java.util.List;

/**
 * 1 to 1 sending
 */
public class LinkedSplit {



    public static void main(String[] args) {
        int taskNum = 15;
        int total = 7352;
        List<TrainSplit.Pair> list = TrainSplit.getTask(taskNum, total);

        System.out.println(list.toString());

        List<TrainSplit> taskList = new ArrayList<>();
        TrainSplit root = null;
        TrainSplit master = null;
        for (int i = 0; i < taskNum; i++) {
            // task
            TrainSplit.Pair fragment = list.get(i);

            TrainSplit task = null;
            if (root == null) {
                // master 1st
                task = new TrainSplit(i, new TrainSplit.Settings(fragment.start, fragment.end), taskNum - 1, true);
                master = task;
            }
            else {
                task = new TrainSplit(root.getQueue(), i, new TrainSplit.Settings(fragment.start, fragment.end), true);
                // last one will not wait for any nodes
                if (i == taskNum - 1) {
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
