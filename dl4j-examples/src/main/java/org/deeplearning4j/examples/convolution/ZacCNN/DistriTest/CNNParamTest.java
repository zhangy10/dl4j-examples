package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.List;
import java.util.Map;

public class CNNParamTest {

    public static void main(String[] args) throws Exception {

        int epoch = 3;

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        int taskNum = 10;

        char delimiter = ',';
        File file = new ClassPathResource("zac.txt").getFile();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        // last pos is label
        // by channel, the label index will be the second one, not actual label index
        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row

        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 4;

        // channel * width = inputwidth
        int channel = 3;
        int height = 1;
        int width = 4;


        HarReader reader = new HarReader(numLinesToSkip, height, width, channel, numClasses, taskNum, delimiter);
        reader.initialize(new FileSplit(file));
        reader.setLabelFromOne(false);

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, batchSize, labelIndex, numClasses);

        // Test for reading data
//        int i = 0;
//        while (iterator.hasNext()) {
//            DataSet data = iterator.next(batchSize);
//            i++;
//            System.out.println("batch " + i + " : ");
//            System.out.println(data.getFeatures());
//            System.out.println(data.getLabels());
//        }

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        iterator.setPreProcessor(normalizer);


//        StandardizeStrategy

        MultiLayerNetwork model = getModel(numClasses, height, width, channel);
        model.fit(iterator, epoch);


//        //evaluate the model on the test set
//        Evaluation eval = new Evaluation(3);
//        INDArray output = model.output(testData.getFeatures());
//        eval.eval(testData.getLabels(), output);
//        log.info(eval.stats());
    }


    private static ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private static SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private static DenseLayer full(String name, int out, double bias) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dist(new NormalDistribution(0, 1)).build();
    }

    public static MultiLayerNetwork getModel(int classNum, int h, int w, int channel) {
        long seed = 6;
        double nonZeroBias = 1; //偏差

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                           .seed(seed)
                                           .activation(Activation.RELU)
                                           .weightInit(WeightInit.DISTRIBUTION)
                                           .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                                           .updater(new Adam(0.1))
                                           .l2(1e-4)
                                           .list()
                                           .layer(0, convNet("c1", channel, 6, new int[]{1, 2}, new int[]{1, 1}, new int[]{0, 0}, 0))
                                           // update padding issue
//                                           .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
                                           .layer(1, maxpooling("m1", new int[]{1, 2}, new int[]{1, 2}))
                                           .layer(2, full("f1", 20, nonZeroBias))
//                                           .layer(3, new DenseLayer.Builder().nIn(6).nOut(9)
//                                                         .build())
                                           .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("o1")
                                                         .activation(Activation.SOFTMAX)
                                                         .nOut(classNum).build())
                                           // will do flat matrix to 2d for full-connected layer
                                           .setInputType(InputType.convolutional(h, w, channel)) // image的长宽
                                           .backprop(true).pretrain(false)
                                           .build();


        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(listener);
        return model;
    }

    private static int epoc = 0;

    private static INDArray w0 = null;

    private static INDArray Aw = null;

    public static TrainingListener listener = new TrainingListener() {
        @Override
        public void iterationDone(Model model, int iteration, int epoch) {

            if (epoc != epoch) {
                // new epoc
                epoc = epoch;
                Aw = null;
            }

            if (Aw == null) {
                Aw = model.gradient().gradient().dup();
            }
            else {
                Aw.addi(model.gradient().gradient());
            }

            System.out.println("iteration done: " + iteration + " model score: " + model.score());
        }

        @Override
        public void onEpochStart(Model model) {
            System.out.println("onEpochStart");
            MultiLayerNetwork network = (MultiLayerNetwork) model;
            Layer l1 = network.getLayer(0);
            Layer l2 = network.getLayer(1);
            Layer l3 = network.getLayer(2);
            Layer l4 = network.getLayer(3);

            w0 = model.params().dup();

//            Updater u = ((MultiLayerNetwork) model).getUpdater();
//            u.update();
        }

        @Override
        public void onEpochEnd(Model model) {
            System.out.println("onEpochEnd");

            INDArray newW = w0.sub(Aw);

            INDArray w = model.params();
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
