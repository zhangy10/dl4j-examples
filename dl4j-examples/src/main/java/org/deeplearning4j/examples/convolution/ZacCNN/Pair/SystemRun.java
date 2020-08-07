package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;

public class SystemRun {

    public static SyncPolicy policy = SyncPolicy.EPOC;
    public static ModelLayer layerConfig = ModelLayer.ONE;

    public static boolean needLowScale = false;
    public static boolean isAlex = false;
    public static boolean isTestRound = true;

    public static List<DataSet.DataType> dataset = new ArrayList<>();
    public static boolean isMaster = true;

    /**
     * training data order
     *
     */
    static {
        dataset.add(DataSet.DataType.EMG);
        dataset.add(DataSet.DataType.HAR);
        dataset.add(DataSet.DataType.FALL);
        dataset.add(DataSet.DataType.MHe);
        dataset.add(DataSet.DataType.OP);
        dataset.add(DataSet.DataType.PAMA);
    }

    public static DataSet.DataType type = DataSet.DataType.HAR;
    public static int dataIndex = 0;

    public static String dataPath = "/Users/zhangyu/GoogleDrive/PHD/All_Papers/MDLdroid/Submission/Final_Data/final_data";
    public static String outputPath = "/Users/zhangyu/Desktop/";

    public static int startTaskNum = 1;
    public static int maxTaskNum = 9;

    public static MDLModel.Type modelType = MDLModel.Type.LENET1D;
    public static boolean isScaleDecay = true;
    public static boolean isMobileNet = false;

    public static void main(String[] args) {
        type = dataset.get(dataIndex);
        // start first task
        next(startTaskNum);
    }

    private static void next(int index) {
//        if (isMaster) {
        new MasterRun(index, type, outputPath, donelistener, modelType).run();
//        }
//        else {
//            new LinkedSplit(index, type, basePath, donelistener).run();
//        }
    }

    private static SplitListener donelistener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            startTaskNum++;
            if (startTaskNum <= maxTaskNum) {
                next(startTaskNum);
            } else {
                System.out.println("\n ================================================================== \n");
                startTaskNum = 1;
                dataIndex++;
                if (dataIndex < dataset.size()) {
                    // will start to next dataset
                    type = dataset.get(dataIndex);
                    // start first task
                    next(startTaskNum);
                }
            }
        }
    };

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
