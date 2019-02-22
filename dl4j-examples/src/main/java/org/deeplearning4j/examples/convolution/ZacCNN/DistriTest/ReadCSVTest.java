package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.deeplearning4j.examples.dataexamples.CSVExample;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.StandardizeStrategy;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;

public class ReadCSVTest {


    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    public static void main(String[] args) throws Exception {

        int epoch = 2;

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        int taskNum = 10;

        char delimiter = ',';
        File file = new ClassPathResource("iris_test.txt").getFile();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        // last pos is label
        // by channel, the label index will be the second one, not actual label index
        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row

        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 2;

        // channel * width = inputwidth
        int channel = 2;
        int height = 1;
        int width = 2;


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

    public static MultiLayerNetwork getModel(int classNum, int h, int w, int channel) {
        long seed = 6;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.1))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nOut(6)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(6).nOut(9)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(9).nOut(classNum).build())
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

    public static TrainingListener listener = new TrainingListener() {
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
