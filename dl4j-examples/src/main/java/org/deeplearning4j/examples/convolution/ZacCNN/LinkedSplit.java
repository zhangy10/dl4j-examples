package org.deeplearning4j.examples.convolution.ZacCNN;


import java.util.ArrayList;
import java.util.List;

/**
 * 1 to 1 sending
 */
public class LinkedSplit {

    private int taskNum;
    private DataType dataType;
    private String basePath;
    private String logPath;
    private String log = "";
    private SplitListener donelistener;

    public LinkedSplit(int taskNum, DataType dataType, String basePath, SplitListener donelistener) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
    }

    public static void main(String[] args) {
        String file = "/Users/zhangyu/Desktop/";
        LinkedSplit runner = new LinkedSplit(2, DataType.TEST, file, null);
        runner.run();
    }


    private SplitListener listener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            // save output to files
            FileUtils.write(log + output, logPath);
            if (donelistener != null) {
                donelistener.trainDone(null);
            }
        }
    };


    public void run() {
        DataType type = dataType;
        int taskN = taskNum;

        String modelFile = basePath + taskNum + "_" + type + "_chain_model.bin";
        logPath = basePath + taskNum + "_" + type + "_chain_log.txt";

        List<TrainSplit.Pair> list = TrainSplit.getTask(taskN, DataSet.getConfig(type).getTaskNum());
        log = list.toString() + "\n";
        System.out.println(log);

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
                task = new TrainSplit(i, config.setTaskRange(fragment.start, fragment.end), taskN - 1, true, listener, modelFile);
                master = task;
            } else {
                task = new TrainSplit(root.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), true, listener, modelFile);
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
