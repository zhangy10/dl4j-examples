package org.deeplearning4j.examples.convolution.ZacCNN.ImageTest;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Random;

import static java.lang.Math.toIntExact;

public class ImageEva {

    static boolean isCF10 = true;

    static int balanceTestSize = 500;

    static int height = 28;
    static int width = 28;
    static int channels = 1; // single channel for grayscale images
    static int outputNum = 6; // 10 digits classification

    static int batchSize = 16;

    public static void main(String[] args) throws Exception{

        Random randNumGen = new Random(1234);

        String file = null;
        if (isCF10) {
            file = "/Users/zhangyu/Downloads/10png/test";
        } else {
            file = "/Users/zhangyu/Downloads/mnist_png/testing";
        }

        File testData = new File(file);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label

        int testSize = toIntExact(testSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, labelMaker, testSize, outputNum, balanceTestSize);
        InputSplit[] testInput = testSplit.sample(pathFilter);

        if (isCF10) {
            height = 32;
            width = 32;
            channels = 3;
        }

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testInput[0]);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler); // same normalization for better results


        File model = new File("/Users/zhangyu/Desktop/" + "model_phone.bin");
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(model);

        Evaluation eval = network.evaluate(testIter);
        System.out.println(eval.stats());
    }
}
