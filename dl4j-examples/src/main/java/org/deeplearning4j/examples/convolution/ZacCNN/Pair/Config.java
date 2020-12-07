package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import java.io.File;

public class Config implements Cloneable {

    private int numLinesToSkip = 0;
    private int taskNum = 7352;

    private char delimiter = ',';
    private int labelIndex = 1;
    private int batchSize = 8;
    private int epoch = 20;

    // channel * width = inputwidth
    private int channel = 9;
    private int height = 1;
    private int width = 128;
    private int numClasses = 6;
    private boolean isNoraml = false;
    private String dataPath = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_train.csv";
    private String testPath = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_test.csv";

    private String modelConfig;

    // model settings
    private double nonZeroBias = 1; //偏差
    private double dropOut = 0.8; //随机丢弃比例
    private long seed = 42;

    private int kernal = 64;
    private int pooling = 2;

    // layer settings
    private int c1_out = 36;
    private int c2_out = 72;
    private int c3_out = 0;
    private int f1_out = 300;
    private int f2_out = 0;
    private int f3_out = 0;

    private double learnRate = 0.001;
    private double gamma = 0.5;

    public Config() {
    }

    public Config(String modelConfig) {
        this.modelConfig = modelConfig;
    }

    /**
     * default data config
     *
     * @param numLinesToSkip
     * @param taskNum
     */
    public Config(int numLinesToSkip, int taskNum) {
        this.numLinesToSkip = numLinesToSkip;
        this.taskNum = taskNum;
    }

    public Config setTaskRange(int numLinesToSkip, int taskNum) {
        this.numLinesToSkip = numLinesToSkip;
        this.taskNum = taskNum;
        return this;
    }

    public Config setDataPath(String path) {
        this.dataPath = path;
        return this;
    }

    public Config setTestPath(String path) {
        this.testPath = path;
        return this;
    }

    public File getFile() throws Exception {
        return new File(dataPath);
    }

    @Override
    public Config clone() {
        try {
            return (Config) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public String getTestPath() {
        return testPath;
    }

    public int getNumLinesToSkip() {
        return numLinesToSkip;
    }

    public int getTaskNum() {
        return taskNum;
    }

    public char getDelimiter() {
        return delimiter;
    }

    public int getLabelIndex() {
        return labelIndex;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpoch() {
        return epoch;
    }

    public int getChannel() {
        return channel;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public boolean isNoraml() {
        return isNoraml;
    }

    public String getDataPath() {
        return dataPath;
    }

    public String getModelConfig() {
        return modelConfig;
    }

    public double getNonZeroBias() {
        return nonZeroBias;
    }

    public double getDropOut() {
        return dropOut;
    }

    public long getSeed() {
        return seed;
    }

    public int getKernal() {
        return kernal;
    }

    public int getPooling() {
        return pooling;
    }

    public int getC1_out() {
        return c1_out;
    }

    public int getC2_out() {
        return c2_out;
    }

    public int getC3_out() {
        return c3_out;
    }

    public int getF1_out() {
        return f1_out;
    }

    public int getF2_out() {
        return f2_out;
    }

    public int getF3_out() {
        return f3_out;
    }

    public double getLearnRate() {
        return learnRate;
    }

    public static class Builder {

        private int c1_out = 36;
        private int c2_out = 72;
        private int f1_out = 300;
        private int kernel = 64;
        private String dataPath;
        private String testPath;
        private boolean isNoraml = false;
        private int width = 128;
        private int numClasses = 6;
        private int channel = 9;
        private int batchSize = 8;
        private int epoch = 20;
        private int taskNum = 7352;

        public Builder() {
        }

        public Builder setC1Out(int c1_out) {
            this.c1_out = c1_out;
            return this;
        }

        public Builder setC2Out(int c2_out) {
            this.c2_out = c2_out;
            return this;
        }

        public Builder setf1Out(int f1_out) {
            this.f1_out = f1_out;
            return this;
        }

        public Builder setDataPath(String dataPath) {
            this.dataPath = dataPath;
            return this;
        }

        public Builder isNormal(boolean isNoraml) {
            this.isNoraml = isNoraml;
            return this;
        }

        public Builder setWidth(int width) {
            this.width = width;
            return this;
        }

        public Builder setNumClass(int numClass) {
            this.numClasses = numClass;
            return this;
        }

        public Builder setChannel(int channel) {
            this.channel = channel;
            return this;
        }

        public Builder setBatch(int batch) {
            this.batchSize = batch;
            return this;
        }

        public Builder setEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }

        public Builder setTaskNum(int taskNum) {
            this.taskNum = taskNum;
            return this;
        }

        public Builder setKernel(int kernel) {
            this.kernel = kernel;
            return this;
        }

        public Builder setTestPath(String testPath) {
            this.testPath = testPath;
            return this;
        }

        public Config build() {
            Config c = new Config();
            c.c1_out = c1_out;
            c.c2_out = c2_out;
            c.f1_out = f1_out;
            c.dataPath = dataPath;
            c.isNoraml = isNoraml;
            c.width = width;
            c.numClasses = numClasses;
            c.channel = channel;
            c.batchSize = batchSize;
            c.epoch = epoch;
            c.taskNum = taskNum;
            c.kernal = kernel;
            c.testPath = testPath;
            return c;
        }
    }
}
