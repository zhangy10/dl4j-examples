package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class ArrayFileTest {

    private static String path = "/Users/zhangyu/Desktop/double_array";

    public static void main(String[] args) throws Exception {

        int numLinesToSkip = 0;
        int taskNum = 7352;

        int labelIndex = 1;

        char delimiter = ',';
        File file = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/Har/x_train.csv");

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

        DataSet line = iterator.next();

        System.out.println("feature: \n" + line.getFeatures());

        // write to file
        File savefile = new File(path);
        if (!savefile.exists()) {
            Nd4j.saveBinary(line.getFeatures(), savefile);
        }

        // read from file
        INDArray array = Nd4j.readBinary(savefile);
        System.out.println("Read from file: \n" + array);

    }


}
