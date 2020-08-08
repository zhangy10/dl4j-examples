package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.deeplearning4j.examples.convolution.ZacCNN.FileUtils;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.TaskSplit;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;

public class MasterRun {

    private int taskNum;
    private DataSet.DataType dataType;
    private MDLModel.Type modelType = MDLModel.Type.LENET1D;
    private String basePath;
    private String logPath;
    private String log = "";
    private SplitListener donelistener;

    public MasterRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener) {
        this(taskNum, dataType, basePath, donelistener, MDLModel.Type.LENET1D);
    }

    public MasterRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener, MDLModel.Type modelType) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
        this.modelType = modelType;
    }

    public void setType(MDLModel.Type modelType) {
        this.modelType = modelType;
    }

    public static void main(String[] args) {
        String file = "/Users/zhangyu/Desktop/";

        MasterRun runner = new MasterRun(1, DataSet.DataType.EMG, file, null);
//        runner.setType(MDLModel.Type.LENET1D);
        runner.setType(MDLModel.Type.MOBILENET_1D);
//        runner.setType(MDLModel.Type.TCN);
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
        int task = taskNum;

        String modelFile = basePath + taskNum + "_" + type + "_" + modelType + "_model.bin";
        logPath = basePath + taskNum + "_" + type + "_" + modelType + "_log.txt";

        // split task
        List<TaskSplit.Pair> list = TaskSplit.getTask(task, DataSet.getConfig(type).getTaskNum());
        log = list.toString() + "\n";
        System.out.println(log);

        // assgin task
        TrainMaster master = null;
        List<TrainMaster> slaveList = new ArrayList<>();

        for (int i = 0; i < task; i++) {
            TaskSplit.Pair fragment = list.get(i);
            Config config = DataSet.getNewConfig(type);
            if (i == 0) {
                master = new TrainMaster(i, config.setTaskRange(fragment.start, fragment.end), task - 1, listener, modelFile);
                master.setModelType(modelType);
            } else {
                TrainMaster slave = new TrainMaster(master.getQueue(), i, config.setTaskRange(fragment.start, fragment.end), listener, modelFile);
                slave.setModelType(modelType);
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }

        // control working number to start task
        GradientInfo.clean();
        int started = 0;
        master.start();
        int endNum = slaveList.size();
        // at least 1 to pair aggregation
//        int endNum = 0;
        System.out.println("[Avaliable Task Number]: " + endNum);
        for (int i = 0; i < endNum; i++) {
            slaveList.get(i).start();
        }
    }

}
