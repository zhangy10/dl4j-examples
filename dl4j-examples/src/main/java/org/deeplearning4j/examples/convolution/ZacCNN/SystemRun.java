package org.deeplearning4j.examples.convolution.ZacCNN;

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


    public static void main(String[] args) {
        // start first task
        next(index);
    }

    private static SplitListener donelistener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            index++;
            if (index <= maxTask) {
                next(index);
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
