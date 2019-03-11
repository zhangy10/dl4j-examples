package org.deeplearning4j.examples.convolution.ZacCNN;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.AnimalsClassification;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DepthWiseHar {


    protected static final Logger log = LoggerFactory.getLogger(AnimalsClassification.class);


    /*
     * randomly get a number
     */
    protected static long seed = 42;
    protected static Random rng = new Random(seed);

    // training size 8:2
//    protected static double splitTrainTest = 0.8;

    // whether save the output model
    protected static boolean save = false;

    // data file path
//    private static final String path = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/New_data/Har/";
    private static final String path = "/Users/zber/Desktop/Data/Har";

    private static final String save_path = "";

    private boolean isSave = true;


    public static void main(String[] args) throws Exception {
        new DepthWiseHar().train();
    }


    public void train() throws Exception {
        int numLinesToSkip = 0;
        int taskNum = 7352;

        int labelIndex = 1;

        char delimiter = ',';
        File file = new File("/Users/zber/Desktop/Data/Har/train/train.csv");

        // data settings
        int epochs = 20;
        int numLabels = 6;
        int batchSize = 16;

        int height = 1;
        int width = 128;
        int channels = 9;


        // loading data
        HarReader reader = new HarReader(numLinesToSkip, height, width, channels, numLabels, taskNum, delimiter);
        reader.initialize(new FileSplit(file));

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, batchSize, labelIndex, numLabels);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        // build net
        MultiLayerNetwork network = mbnet(channels, numLabels, height, width);
        network.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        // set listener, 参数有更新 则回调
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

        network.setListeners(listener);

        // training
        network.fit(iterator, epochs);


        // testing


        // save model


    }

    /**
     * Build a layer
     *
     * @param name
     * @param in
     * @param out
     * @param kernel
     * @param stride
     * @param pad
     * @param bias
     * @param
     * @return
     */
    private ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == 0) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }


    private ConvolutionLayer depthWiseNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias, int multiplier) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == 0) {
              return new SeparableConvolution2D.Builder(kernel,stride,pad).name(name).nOut(out).biasInit(bias).depthMultiplier(multiplier).build();
//            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).build();
        }

        DepthwiseConvolution2D d = new DepthwiseConvolution2D();
          return new SeparableConvolution2D.Builder(kernel,stride,pad).name(name).nIn(in).nOut(out).biasInit(bias).depthMultiplier(multiplier).build();
//        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }


    private SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private DenseLayer full(String name, int out, double bias, double dropOut) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }

    /**
     * MB Net
     *
     * @return
     */
    public MultiLayerNetwork mbnet(int channels, int numLabels, int height, int width) {
        // ??
        double nonZeroBias = 1; //偏差
        double dropOut = 0.8; //随机丢弃比例

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.NORMAL) //根据给定的分布采样参数
//                .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
            .activation(Activation.RELU)
            .updater(new Adam(0.001))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            //采用除以梯度2范数来规范化梯度防止梯度消失或突变
            .l2(5 * 1e-4)
            .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
//            .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
            .layer(0, depthWiseNet("d1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0,8))
            .layer(1, maxpooling("m1", new int[]{1, 2}, new int[]{1, 2}))
            .layer(2, depthWiseNet("d2", 0, 72, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 16}, nonZeroBias,8))
            .layer(3, maxpooling("m2", new int[]{1, 2}, new int[]{1, 2}))
            .layer(4, full("f1", 300, nonZeroBias, dropOut))
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .name("o1")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();
        return new MultiLayerNetwork(conf);
    }

    public TrainingListener listener = new TrainingListener() {
        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            System.out.println("iteration done: " + iteration + " model score: " + model.score());
        }

        @Override
        public void onEpochStart(Model model) {
            System.out.println("onEpochStart");
        }

        @Override
        public void onEpochEnd(Model model) {
            System.out.println("onEpochEnd");
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {
            System.out.println("onForwardPass");
        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {
            System.out.println("onForwardPass");
        }

        @Override
        public void onGradientCalculation(Model model) {
            System.out.println("onGradientCalculation");
        }

        @Override
        public void onBackwardPass(Model model) {
            System.out.println("onBackwardPass");
        }
    };
}
