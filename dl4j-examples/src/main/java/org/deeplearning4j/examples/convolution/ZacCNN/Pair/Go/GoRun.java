package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base.BaseRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.GradientInfo;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GoRun extends BaseRun {

    // assgin task
    private static List<TrainGo> taskList = new ArrayList<>();
    private static Random random = new Random();

    public static TrainGo getRandomTarget(int request) {
        TrainGo target = null;
        if (taskList.size() > 1) {
            synchronized (taskList) {
                int sendTo = -1;
                while (true) {
                    sendTo = random.nextInt(taskNum);
                    if (sendTo != request) {
                        break;
                    }
                }
                if (sendTo != -1) {
                    target = taskList.get(sendTo);
                }
            }
        }
        return target;
    }

    public static int getNodeNum() {
        return taskList.size();
    }

    private static void cleanList() {
        synchronized (taskList) {
            taskList.clear();
        }
    }

    public GoRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener) {
        this(taskNum, dataType, basePath, donelistener, MDLModel.Type.LENET1D);
    }

    public GoRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener, MDLModel.Type modelType) {
        super(taskNum, dataType, basePath, donelistener, modelType);
    }

    public static void main(String[] args) throws Exception {
        String file = "/Users/zhangyu/Desktop/";

        GoRun runner = new GoRun(9, DataSet.DataType.HAR, file, null);
        runner.setType(MDLModel.Type.LENET1D);
//        runner.setType(MDLModel.Type.MOBILENET_1D);
//        runner.setType(MDLModel.Type.TCN);
        runner.run();
    }

    @Override
    protected void start() {
        cleanList();
        // init goscale
        double goscale = 1 / (double) taskNum;
        // get task
        List<Config> configs = getConfig();

        // assgin task
        TrainGo master = null;

        for (int i = 0; i < taskNum; i++) {
            Config config = configs.get(i);
            if (i == 0) {
                master = new TrainGo(i, config, taskNum - 1, listener, modelFile, goscale);
                master.setModelType(modelType);
                master.setExistingFile(existingFile);
                taskList.add(master);
            } else {
                TrainGo slave = new TrainGo(master.getQueue(), i, config, listener, modelFile, goscale);
                slave.setModelType(modelType);
                slave.setExistingFile(existingFile);
                master.addSlave(slave.getQueue());
                taskList.add(slave);
            }
        }

        // control working number to start task
        GradientInfo.clean();

        for (int i = 0; i < taskList.size(); i++) {
            taskList.get(i).start();
            if (i != 0) {
                master.addAvaliable();
            }
        }
        // start all tasks
        System.out.println("[Avaliable Task Number]: " + master.getAvaliableNum() + ", All task: " + taskList.size());
    }
}
