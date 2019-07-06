package org.deeplearning4j.examples.convolution.ZacCNN;

public class SystemRun {

    static DataType type = DataType.TEST;
    static boolean isMaster = true;
    static int maxTask = 3;
    static String basePath = "/Users/zhangyu/Desktop/";
    static int index = 1;


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


    // sync frequency
    public static int getSyncFrequency(Config config, SyncPolicy policy) {
        if (config != null) {
            int batchNum = (int) Math.ceil(config.getTaskNum() / (float) config.getBatchSize());
            switch (policy) {
                case EPOC:
                    return batchNum;
                case HALF_EPOC:

                    break;
                case QUART_EPOC:

                    break;
                case BATCH:

                    break;
            }
        }
        return 0;
    }

    public enum SyncPolicy {
        EPOC,
        HALF_EPOC,
        QUART_EPOC,
        BATCH
    }
}
