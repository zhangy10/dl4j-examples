package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base.BaseRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.GradientInfo;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;

public class MainRun extends BaseRun {

    public MainRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener) {
        this(taskNum, dataType, basePath, donelistener, MDLModel.Type.LENET1D);
    }

    public MainRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener, MDLModel.Type modelType) {
        super(taskNum, dataType, basePath, donelistener, modelType);
    }

    public static void main(String[] args) throws Exception {
        String file = "/Users/zhangyu/Desktop/";

        MainRun runner = new MainRun(9, DataSet.DataType.HAR, file, null);
        runner.setType(MDLModel.Type.LENET1D);
//        runner.setType(MDLModel.Type.MOBILENET_1D);
//        runner.setType(MDLModel.Type.TCN);
        runner.run();

        // Test training time
//        new ReadModelTest(DataSet.getConfig(DataSet.DataType.HAR), MDLModel.Type.LENET1D, 100).runTask();
    }

    @Override
    protected void start() {
        List<Config> configs = getConfig();
        // assgin task
        TrainMain master = null;
        List<TrainMain> slaveList = new ArrayList<>();

        for (int i = 0; i < taskNum; i++) {
            Config config = configs.get(i);
            if (i == 0) {
                master = new TrainMain(i, config, taskNum - 1, listener, modelFile);
                master.setModelType(modelType);
                master.setExistingFile(existingFile);
            } else {
                TrainMain slave = new TrainMain(master.getQueue(), i, config, listener, modelFile);
                slave.setModelType(modelType);
                slave.setExistingFile(existingFile);
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }

        // control working number to start task
        GradientInfo.clean();
//        int started = 0;
        master.start();
        int endNum = slaveList.size();
        // at least 1 to pair aggregation
//        int endNum = 7;
        System.out.println("[Avaliable Task Number]: " + endNum);
        for (int i = 0; i < endNum; i++) {
            slaveList.get(i).start();
            master.addAvaliable();
        }
    }
}

