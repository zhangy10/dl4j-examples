package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * Animal Classification
 * <p>
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 * <p>
 * References:
 * - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 * - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 * <p>
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 * - Add additional images to the dataset
 * - Apply more transforms to dataset
 * - Increase epochs
 * - Try different model configurations
 * - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class AnimalsClassification {
    protected static final Logger log = LoggerFactory.getLogger(AnimalsClassification.class);


    /*
     * Input data is 4 classes, and each class has 20 images, so 4 * 20 = 80
     *
     * image: 100 * 100, RGB 3 channels
     *
     */
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;

    // the number of each process round, and all 80 inputs will be processed in 4 rounds
    protected static int batchSize = 20;

    /*
     * randomly get a number
     */
    protected static long seed = 42;
    protected static Random rng = new Random(seed);

    /*
     * training length, if it's long, then train time will be long
     */
    protected static int epochs = 50;

    // training size 8:2
    protected static double splitTrainTest = 0.8;

    // whether save the output model
    protected static boolean save = false;

    // for balance input data
    protected static int maxPathsPerLabel = 20;

    /*
     * model choose or design a new one
     */
    protected static String modelType = "LeNet"; // LeNet, AlexNet or Custom but you need to fill it out
    // the number of classes
    private int numLabels;

    private static final String path = "dl4j-examples/src/main/resources/animals/";
    private static final String rootDir = "dl4j-examples";
    private static final String errorDir = "deeplearning4j-examples";


    /**
     * Loading data
     *
     * @return
     * @throws Exception
     */
    private Data loading() throws Exception {

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        // can get the dir name as label string
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        String dir = System.getProperty("user.dir");
        dir = dir.replace(errorDir, rootDir);

        File mainPath = new File(dir, path);

        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);

        // get all file numbers: 22, 20, 21, 20 = 83
        int numExamples = toIntExact(fileSplit.length());

        // 4 classes
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.

        // each class will have maxPathsPerLabel files 20 and in total 80 out of 83
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);


        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        // 64
        InputSplit trainData = inputSplit[0];
        // 16
        InputSplit testData = inputSplit[1];


        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         *
         *
         *  transform an image to an array data
         **/

        // flip (翻转) an image by a random number
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(warpTransform, 0.5));

        // get 3 channel data into array
        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        return new Data(labelMaker, inputSplit, transform);
    }

    private class Data {
        ParentPathLabelGenerator labelMaker = null;
        InputSplit trainData = null;
        InputSplit testData = null;
        ImageTransform transform = null;

        public Data(ParentPathLabelGenerator labelMaker, InputSplit[] data, ImageTransform transform) {
            this.labelMaker = labelMaker;
            this.trainData = data[0];
            this.testData = data[1];
            this.transform = transform;
        }
    }


    /**
     * Building a net work and training
     *
     * @param args
     * @throws Exception
     */
    public void run(String[] args) throws Exception {

        Data input = loading();

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         *
         *
         *  change all data to the range from 0 to 1 (normalization)
         **/

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
//        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = lenetModel();
                break;
            case "AlexNet":
                network = alexnetModel();
                break;
            case "custom":
                network = customModel();
                break;
            case "mb":
                network = mbnet();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        network.init();

        // network.setListeners(new ScoreIterationListener(listenerFreq));

        // if updated by each round, then print the score
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        // set listener, 参数有更新 则回调
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));


        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/



        log.info("Train model....");

        // 图片读取器: 128, 1, 9, file path
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, input.labelMaker);

        // Train without transformations, 训练不翻转的数据，目的是跑一遍初始化参数
        recordReader.initialize(input.trainData, null);

        // 建立数据迭代器，输入，每次处理多少，class数量...
        // 数据迭代器
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        // 收集规范化统计信息
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        scaler.fit(dataIter);
        // 对数据进行规范化
        dataIter.setPreProcessor(scaler);

        // training
        network.fit(dataIter, epochs);

        // Train with transformations, 训练翻转的数据，有了初始化参数再上各种翻转数据
        recordReader.initialize(input.trainData, input.transform);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        // 再次规范化统计？？
        scaler.fit(dataIter);
        // 再次数据规范化？？
        dataIter.setPreProcessor(scaler);

        // training
        network.fit(dataIter, epochs);





        log.info("Evaluate model....");
        // 处理数据一套流程，数据迭代器，规范化统计，规范化
        recordReader.initialize(input.testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));


        // 测试分类器
        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
        dataIter.reset();
        DataSet testDataSet = dataIter.next();

        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);

        // 分类...
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }
        log.info("****************Example finished********************");
    }


    // Build a layer


    private ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == 0) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private DenseLayer full(String name, int out, double bias, double dropOut) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }

    /*
     *
     *
     * *******************************
     */

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        //3*3的卷积层，卷积核大小3*3，步副大小1*1，补充边框1*1
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        //5*5的卷积层，卷积核大小5*5
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        //子采样层，本例中卷积核大小是2*2，步副2*2
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    // Build a network

    public MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)  // ??
                .activation(Activation.RELU)
                .weightInit(WeightInit.NORMAL) //参数服从均值为0，方差为2.0/(fanIn + fanOut)的高斯分布，fanIn是上一层节点数，fanOut是当前层节点数
                .updater(new Nesterovs(0.0001, 0.9))  //采用可变学习率，动量衰减参数为0.9的参数优化方法
                .list() //list代表多层网络，0,1,2,3,4层已经介绍过，5层是输出层
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //使用交叉熵作为损失函数
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                    .setInputType(InputType.convolutional(height, width, channels)) // image的长宽
                .build();

        return new MultiLayerNetwork(conf);

    }


    /**
     * Test
     *
     * @return
     */
    public MultiLayerNetwork mbnet() {

        double nonZeroBias = 1; //偏差
        double dropOut = 0.8; //随机丢弃比例

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.NORMAL) //根据给定的分布采样参数
                .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
                .activation(Activation.RELU)
//                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .updater(new Adam(0.001))

//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                //采用除以梯度2范数来规范化梯度防止梯度消失或突变
//                .l2(5 * 1e-4)
                .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                .layer(0, convNet("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, maxpooling("maxp1", new int[]{3, 3}, new int[]{2, 2}))
                .layer(2, convNet("cnn2", 0, 256, new int[]{1, 1}, new int[]{2, 2}, new int[]{1, 1}, nonZeroBias))
                .layer(3, maxpooling("maxp2", new int[]{3, 3}, new int[]{2, 2}))
                .layer(4, full("f1", 4096, nonZeroBias, dropOut))
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();


//                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
//                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
//                .layer(2, maxPool("maxpool1", new int[]{3, 3}))
//                .layer(3, conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
//                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
//                .layer(5, maxPool("maxpool2", new int[]{3, 3}))
//                .layer(6, conv3x3("cnn3", 384, 0))
//                .layer(7, conv3x3("cnn4", 384, nonZeroBias))
//                .layer(8, conv3x3("cnn5", 256, nonZeroBias))
//                .layer(9, maxPool("maxpool3", new int[]{3, 3}))
//                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
//                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
//                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .name("output")
//                        .nOut(numLabels)
//                        .activation(Activation.SOFTMAX)
//                        .build())
//                .backprop(true)
//                .pretrain(false)
//                .setInputType(InputType.convolutional(height, width, channels))
//                .build();

        return new MultiLayerNetwork(conf);
    }


    public MultiLayerNetwork mbnetDepth() {


        return null;
    }

    public MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1; //偏差
        double dropOut = 0.5; //随机丢弃比例

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION) //根据给定的分布采样参数
                .dist(new NormalDistribution(0.0, 0.01)) //均值为0，方差为0.01的正态分布
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                .l2(5 * 1e-4)
                .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3, 3}))
                .layer(3, conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3, 3}))
                .layer(6, conv3x3("cnn3", 384, 0))
                .layer(7, conv3x3("cnn4", 384, nonZeroBias))
                .layer(8, conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3, 3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork customModel() {
        /**
         * Use this method to build your own custom model.
         **/
        return null;
    }

    public static void main(String[] args) throws Exception {
        new AnimalsClassification().run(args);
    }

}
