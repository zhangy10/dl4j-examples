package org.deeplearning4j.examples.convolution.ZacCNN;

import java.util.ArrayList;
import java.util.List;

public class MasterSplit {

    private int taskNum;
    private DataType dataType;
    private String basePath;
    private String logPath;
    private String log = "";
    private SplitListener donelistener;

    public MasterSplit(int taskNum, DataType dataType, String basePath, SplitListener donelistener) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
    }

    public static void main(String[] args) {
        String file = "/Users/zhangyu/Desktop/";
        MasterSplit runner = new MasterSplit(2, DataType.MHe, file, null);
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
        int task = taskNum;

        String modelFile = basePath + taskNum + "_" + type + "_model.bin";
        logPath = basePath + taskNum + "_" + type + "_log.txt";

        // split task
        List<TrainSplit.Pair> list = TrainSplit.getTask(task, DataSet.getConfig(type).getTaskNum());
        log = list.toString() + "\n";
        System.out.println(log);

        // assgin task
        TrainSplit master = null;
        List<TrainSplit> slaveList = new ArrayList<>();
        for (int i = 0; i < task; i++) {
            TrainSplit.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            if (i == 0) {
                master = new TrainSplit(i, config.setTaskRange(fragment.start, fragment.end), task - 1, listener, modelFile);
            } else {
                TrainSplit slave = new TrainSplit(master.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), listener, modelFile);
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
