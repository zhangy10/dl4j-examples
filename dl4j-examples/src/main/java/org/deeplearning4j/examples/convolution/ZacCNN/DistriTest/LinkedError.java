package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.deeplearning4j.examples.convolution.ZacCNN.*;

import java.util.ArrayList;
import java.util.List;

public class LinkedError {

    private int taskNum;
    private DataType dataType;
    private String basePath;
    private String logPath;
    private String log = "";
    private SplitListener donelistener;

    public LinkedError(int taskNum, DataType dataType, String basePath, SplitListener donelistener) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
    }

    public static void main(String[] args) {
        String file = "/Users/zhangyu/Desktop/";
        LinkedError runner = new LinkedError(9, DataType.HAR, file, null);
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

        List<LinkTrain.Pair> list = LinkTrain.getTask(taskN, DataSet.getConfig(type).getTaskNum());
        log = list.toString() + "\n";
        System.out.println(log);

        List<LinkTrain> taskList = new ArrayList<>();
        LinkTrain root = null;
        LinkTrain master = null;
        for (int i = 0; i < taskN; i++) {
            // task
            LinkTrain.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            LinkTrain task = null;
            if (root == null) {
                // master 1st
                task = new LinkTrain(i, config.setTaskRange(fragment.start, fragment.end), taskN - 1, true, listener, modelFile);
                master = task;
            } else {
                task = new LinkTrain(root.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), true, listener, modelFile);
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

        for (LinkTrain t : taskList) {
            t.start();
        }
    }


}
