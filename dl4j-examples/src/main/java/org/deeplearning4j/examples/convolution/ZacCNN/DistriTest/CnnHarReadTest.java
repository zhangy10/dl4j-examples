package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;

public class CnnHarReadTest {


    public static void main(String[] args) throws Exception{

        int numLinesToSkip = 0;
        int taskNum = 7352;

        int labelIndex = 1;

        char delimiter = ',';
        File file = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/x_train.csv");

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

        // loading test
        // testing
        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/test.csv");
        reader.initialize(new FileSplit(testFile));

        iterator = new RecordReaderDataSetIterator(reader, batchSize, labelIndex, numLabels);
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        System.out.println("test and train done...");

        int i = 0;
        while(iterator.hasNext()) {
            DataSet data = iterator.next();
            System.out.println(++i);
        }

    }
}
