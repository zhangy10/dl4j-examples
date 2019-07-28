package org.deeplearning4j.examples.convolution.ZacCNN;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.List;

public class SystemRun {

    static DataType type = DataType.FALL;
    static boolean isMaster = true;
    static int maxTask = 3;
    static String basePath = "/Users/zhangyu/Desktop/";
    static int index = 1;


    public static SyncPolicy policy = SyncPolicy.EPOC;
    public static ModelLayer layerConfig = ModelLayer.ONE;

    public static boolean needLowScale = false;
    public static boolean isAlex = false;
    public static boolean isTestRound = true;

    public static List<DataType> dataset = new ArrayList<>();
    public static int dataIndex = 0;

    static {
//        dataset.add(DataType.EMG);
//        dataset.add(DataType.FALL);
//        dataset.add(DataType.HAR);
//        dataset.add(DataType.MHe);
//        dataset.add(DataType.OP);
//        dataset.add(DataType.PAMA);

        // for test
        dataset.add(DataType.TEST);
        dataset.add(DataType.TEST2);
        dataset.add(DataType.TEST3);
        dataset.add(DataType.TEST4);
    }

    public static void main(String[] args) {
        type = dataset.get(dataIndex);
        // start first task
        next(index);
    }

    private static SplitListener donelistener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            index++;
            if (index <= maxTask) {
                next(index);
            } else {
                System.out.println("\n ================================================================== \n");
                index = 1;
                dataIndex++;
                if (dataIndex < dataset.size()) {
                    // will start to next dataset
                    type = dataset.get(dataIndex);
                    // start first task
                    next(index);
                }
            }
        }
    };

    private static void next(int index) {
        if (isMaster) {
            new MasterSplit(index, type, basePath, donelistener).run();
        } else {
            new LinkedSplit(index, type, basePath, donelistener).run();
        }
    }

    public enum SyncPolicy {
        EPOC,
        HALF_EPOC(2),
        // TODO issues...
//        QUART_EPOC(4),
        BATCH;

        private int syncNum = 0;

        SyncPolicy() {
        }

        SyncPolicy(int syncNum) {
            this.syncNum = syncNum;
        }

        public int getSyncNum() {
            return syncNum;
        }
    }

    public enum ModelLayer {
        ONE, TWO, THREE, LSTM
    }
}
