package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.SystemRun;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NonIIDRun extends SystemRun {

    public static int bound = 100;
    public static int taskNum = 9;

    public static int testTimes = 5;
    public static int taskState = 4;

    public static int time = 0;
    public static int state = 0;
    public static int seed = 123;
    public static List<Integer> pastSeed = new ArrayList<>();

    public NonIIDRun() {
        isIID = false;
        dataset.clear();
        dataset.add(DataSet.DataType.EMG);
//        dataset.add(DataSet.DataType.HAR);
//        dataset.add(DataSet.DataType.MHe);

        type = dataset.get(dataIndex);
        updateSeed();
        next(state);
    }

    public static void main(String[] args) {
        new NonIIDRun();
    }

    private static void updateSeed() {
        while (true) {
            seed = new Random().nextInt(bound);
            if (!pastSeed.contains(seed)) {
                pastSeed.add(seed);
                System.out.print("(-----------------[" + pastSeed + "]-----------------------)");
                break;
            }
        }
    }

    protected static void next(int state) {
        switch (state) {
            case 0:
                // v2
                SystemRun.isScaleDecay = true;
                SystemRun.policy = SyncPolicy.EPOC;
                new MainRun(taskNum, type, outputPath, donelistener, modelType).setRandomSeed(seed).setTag("V2_r" + time).run();
                break;
            case 1:
                // v1
                SystemRun.isScaleDecay = false;
                SystemRun.policy = SyncPolicy.EPOC;
                new MainRun(taskNum, type, outputPath, donelistener, modelType).setRandomSeed(seed).setTag("V1_r" + time).run();
                break;
            case 2:
                // go
                SystemRun.policy = SyncPolicy.EPOC;
                new GoRun(taskNum, type, outputPath, donelistener, modelType).setRandomSeed(seed).setTag("Go_r" + time).run();
                break;
            case 3:
                // go b
                SystemRun.policy = SyncPolicy.Bernoulli;
                new GoRun(taskNum, type, outputPath, donelistener, modelType).setRandomSeed(seed).setTag("Go B_r" + time).run();
                break;
        }
    }

    private static SplitListener donelistener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            System.out.println("\n ================================================================== \n");
            state++;
            if (state < taskState) {
                // switch state
                next(state);
            } else {
                time++;
                state = 0;
                if (time < testTimes) {
                    // switch time
                    updateSeed();
                    next(state);
                } else {
                    // switch tasknum
                    taskNum++;
                    time = 0;
                    if (taskNum <= maxTaskNum) {
                        updateSeed();
                        next(state);
                    } else {
                        // switch dataset
                        dataIndex++;
                        if (dataIndex < dataset.size()) {
                            // will start to next dataset
                            type = dataset.get(dataIndex);
                            updateSeed();
                            next(state);
                        }
                    }
                }
            }
        }
    };

}
