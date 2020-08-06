package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.deeplearning4j.examples.convolution.ZacCNN.FileUtils;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;

public class ChainRun {


    private int taskNum;
    private DataSet.DataType dataType;
    private String basePath;
    private String logPath;
    private String log = "";
    private SplitListener donelistener;

    public ChainRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
    }

    public static void main(String[] args) {
        String file = "/Users/zhangyu/Desktop/";
        ChainRun runner = new ChainRun(6, DataSet.DataType.HAR, file, null);
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
        DataSet.DataType type = dataType;
        int taskN = taskNum;

        String modelFile = basePath + taskNum + "_" + type + "_chain_model.bin";
        logPath = basePath + taskNum + "_" + type + "_chain_log.txt";

        List<TrainChain.Pair> list = TrainChain.getTask(taskN, DataSet.getConfig(type).getTaskNum());
        log = list.toString() + "\n";
        System.out.println(log);

        List<TrainChain> taskList = new ArrayList<>();
        TrainChain root = null;
        TrainChain master = null;
        for (int i = 0; i < taskN; i++) {
            // task
            TrainChain.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            TrainChain task = null;
            if (root == null) {
                // master 1st
                task = new TrainChain(i, config.setTaskRange(fragment.start, fragment.end), taskN - 1, true, listener, modelFile);
                master = task;
            } else {
                task = new TrainChain(root.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), true, listener, modelFile);
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

        for (TrainChain t : taskList) {
            t.start();
        }
    }
}
