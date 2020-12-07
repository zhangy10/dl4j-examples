package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.GoRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.util.ArrayList;
import java.util.List;

public class SystemRun {

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
    public static String outputPath = "/Users/zhangyu/Desktop/test/";

    public static int startTaskNum = 1;
    public static int maxTaskNum = 9;

    // IID test or NonIID test
    public static boolean isIID = true;
    // set different sync policy, e.g., Bernoulli, EPOC
    public static SyncPolicy policy = SyncPolicy.EPOC;
    public static MDLModel.Type modelType = MDLModel.Type.LENET1D;
    // switch v1 to v2
    public static boolean isScaleDecay = false;
    // switch to mobilenet or tcn
    public static boolean isMobileNet = false;

    public static void main(String[] args) {
        type = dataset.get(dataIndex);
        // start first task
        next(startTaskNum);
    }

    protected static void next(int index) {
//        if (isMaster) {
        // v1, v2
//        new MainRun(index, type, outputPath, donelistener, modelType).run();
        // GoSGD
        new GoRun(index, type, outputPath, donelistener, modelType).run();
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
        BATCH,
        Bernoulli;

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
