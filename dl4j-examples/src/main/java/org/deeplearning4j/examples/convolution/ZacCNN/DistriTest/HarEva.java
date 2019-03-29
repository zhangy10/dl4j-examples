package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;

public class HarEva {


    public static void main(String[] args) throws Exception {

        // read a model
//        File model = new File("/Users/zhangyu/Desktop/" + "model.bin");
//        File model = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/" + "model.bin");
//        File model = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/nor_shuffle_data/" + "model.bin");
        File model = new File("/Users/zhangyu/Desktop/" + "multi_model.bin");
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(model);

        // testing
//        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/nor_shuffle_data/1_test.csv");
//        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/test.csv");
        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_test.csv");

        HarReader reader = new HarReader(0, 1, 128, 9, 6, 9999, ',');
        reader.initialize(new FileSplit(testFile));

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, 16, 1, 6);

//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(iterator);
//        iterator.setPreProcessor(normalizer);

        Evaluation eval = network.evaluate(iterator);

        System.out.println(eval.stats(true));
    }

}
