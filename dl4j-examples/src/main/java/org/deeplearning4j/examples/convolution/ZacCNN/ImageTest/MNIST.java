package org.deeplearning4j.examples.convolution.ZacCNN.ImageTest;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.ZacCNN.TrainSplit;
import org.deeplearning4j.examples.convolution.mnist.MnistClassifier;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.concurrent.BlockingQueue;

import static java.lang.Math.toIntExact;

public class MNIST {

    private static final String basePath = "/Users/zhangyu/Desktop";

    // train should be: 7400
    // test should be: 3000

    // train og:
    // test og:

    static int seed = 1234;
    static int height = 28;
    static int width = 28;
    static int channels = 1; // single channel for grayscale images
    static int outputNum = 10; // 10 digits classification

    static int batchSize = 16;
    static int nEpochs = 20;
//    static int iterations = 1;

    static int balanceTrainSize = 700;
    static int balanceTestSize = 300;

    public static void main(String[] args) throws Exception {

        long start = System.currentTimeMillis();
        Random randNumGen = new Random(seed);

//        log.info("Data load and vectorization...");
//        String localFilePath = basePath + "/mnist_png.tar.gz";
//        if (DataUtilities.downloadFile(dataUrl, localFilePath))
//            log.debug("Data downloaded from {}", dataUrl);
//        if (!new File(basePath + "/mnist_png").exists())
//            DataUtilities.extractTarGz(localFilePath, basePath);

        // Preprocess image data:
        // Train: vectorization of train data
        File trainData = new File("/Users/zhangyu/Downloads/mnist_png/training");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label

        // Test: vectorization of test data
        File testData = new File("/Users/zhangyu/Downloads/mnist_png/testing");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // balance
        int trainSize = toIntExact(trainSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, labelMaker, trainSize, outputNum, balanceTrainSize);
        InputSplit[] trainInput = trainSplit.sample(pathFilter);

        int testSize = toIntExact(testSplit.length());
        pathFilter = new BalancedPathFilter(randNumGen, labelMaker, testSize, outputNum, balanceTestSize);
        InputSplit[] testInput = testSplit.sample(pathFilter);

        // ready to load data
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainInput[0]);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testInput[0]);


        // Ready to load image data
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

        // Normalization:  pixel values from 0-255 to 0-1 (min-max scaling )
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);

        // Set Normalization Op
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler); // same normalization for better results

        long process = System.currentTimeMillis();

        System.out.println("Network configuration and training...");

        // Training:
//        MultiLayerConfiguration conf = lenet();
        MultiLayerConfiguration conf = alexnet();
//        MultiLayerConfiguration conf = alexnet2();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(listener);

        System.out.println("Total num of params: " + net.numParams());

        net.fit(trainIter, nEpochs);

        long end = System.currentTimeMillis();

        // save....
        ModelSerializer.writeModel(net, new File(basePath + "/minist-model.zip"), true);

        System.out.println("Model Total num of params: " + net.numParams());
        System.out.println("Save model done!!");
        System.out.println("Preprocessing Total time: " + (process - start) / 1000);
        System.out.println("Train Total time: " + (end - process) / 1000);

        // output: each epoc average loss value
        List<Double> averageList = new ArrayList<>();
        Iterator<Integer> it = epocLoss.keySet().iterator();
        while (it.hasNext()) {
            int index = it.next();
            List<Double> epocList = epocLoss.get(index);
            int size = epocList.size();
            double average = 0;
            for (int i = 0; i < size; i++) {
                average += epocList.get(i);
            }
            averageList.add(average / size);
        }

        System.out.println(averageList);

        // evaluation while training (the score should go down)
        // Evaluate model:
        Evaluation eval = net.evaluate(testIter);
        System.out.println(eval.stats());
    }


    private static MultiLayerConfiguration lenet() {
        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
//                   .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                   .updater(new Adam(0.001))
//                   .weightInit(WeightInit.XAVIER)
                   .weightInit(WeightInit.DISTRIBUTION)
                   .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                   .activation(Activation.RELU)
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list()
                   .layer(0, convNet("c1", channels, 20, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(1, maxpooling("m1", new int[]{2, 2}, new int[]{2, 2}))
                   .layer(2, convNet("c2", -1, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(3, maxpooling("m1", new int[]{2, 2}, new int[]{2, 2}))
                   .layer(4, full("f1", 500, 0, 0.8))
                   .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                 .name("o1")
                                 .nOut(outputNum)
                                 .activation(Activation.SOFTMAX)
                                 .build())
                   .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                   .build();
    }


    //------------------------------------------------

    private static ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private static SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private static DenseLayer full(String name, int out, double bias, double dropOut) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }

    public static MultiLayerConfiguration alexnet() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1; //偏差
        double dropOut = 0.5; //随机丢弃比例

        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
                   .weightInit(WeightInit.DISTRIBUTION) //根据给定的分布采样参数
                   .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
                   .activation(Activation.RELU)
                   .updater(new Adam(0.001))
//                                           .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
//                                           .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，
                   // 参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                   .layer(0, convNet("c1", channels, 20, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                   .layer(2, maxpooling("m1", new int[]{2, 2}, new int[]{2, 2}))
                   .layer(3, convNet("c2", -1, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                   .layer(5, maxpooling("m2", new int[]{2, 2}, new int[]{2, 2}))
                   .layer(6, convNet("c3", -1, 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(7, convNet("c4", -1, 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(8, convNet("c5", -1, 50, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(9, maxpooling("m3", new int[]{2, 2}, new int[]{2, 2}))
                   .layer(10, full("f1", 500, nonZeroBias, dropOut))
                   .layer(11, full("f2", 500, nonZeroBias, dropOut))
                   .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                  .name("o1")
                                  .nOut(outputNum)
                                  .activation(Activation.SOFTMAX)
                                  .build())
                   .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                   .build();
    }



    public static MultiLayerConfiguration alexnet2() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1; //偏差
        double dropOut = 0.5; //随机丢弃比例

        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
                   .weightInit(WeightInit.DISTRIBUTION) //根据给定的分布采样参数
                   .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
                   .activation(Activation.RELU)
                   .updater(new Adam(0.001))
//                                           .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
//                                           .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，
                   // 参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                   .layer(0, convNet("c1", channels, 96, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                   .layer(2, maxpooling("m1", new int[]{3, 3}, new int[]{2, 2}))
                   .layer(3, convNet("c2", -1, 256, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                   .layer(5, maxpooling("m2", new int[]{3, 3}, new int[]{2, 2}))
                   .layer(6, convNet("c3", -1, 384, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   .layer(7, convNet("c4", -1, 384, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(8, convNet("c5", -1, 256, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
                   .layer(9, maxpooling("m3", new int[]{3, 3}, new int[]{2, 2}))
                   .layer(10, full("f1", 4096, nonZeroBias, dropOut))
                   .layer(11, full("f2", 4096, nonZeroBias, dropOut))
                   .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                  .name("o1")
                                  .nOut(outputNum)
                                  .activation(Activation.SOFTMAX)
                                  .build())
                   .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                   .build();
    }

    private static int epoc;

    private static Map<Integer, List<Double>> epocLoss = new TreeMap<>();

    private static long bstart;


    private static TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                long bend = System.currentTimeMillis();
                long time = bend - bstart;
                bstart = bend;
                System.out.println("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch +
                                       " learning rate: " + network.getLearningRate(0) + " time: " + time);
//                System.out.println("Last Time: " + network.getLastEtlTime());
//                try {
//                    Thread.sleep(1000);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());
        }

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();
        }

        @Override
        public void onEpochEnd(Model model) {
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {

        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {

        }

        @Override
        public void onGradientCalculation(Model model) {

        }

        @Override
        public void onBackwardPass(Model model) {

        }
    };


}
