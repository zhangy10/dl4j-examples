package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.ZacCNN.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.DataType;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;

public class HarEva {


    public static void main(String[] args) throws Exception {

        boolean isPhone = true ;
        // read a model
        File model = null;
        if (isPhone) {
            model = new File("/Users/zhangyu/Desktop/" + "model_phone.bin");
        } else {
            model = new File("/Users/zhangyu/Desktop/" + "multi_model.bin");
        }

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(model);

        DataType type = DataType.OP;
        Config config = DataSet.getConfig(type);

        File testFile = new File(config.getTestPath());

        HarReader reader = new HarReader(config.getNumLinesToSkip(), config.getHeight(), config.getWidth(), config.getChannel(),
            config.getNumClasses(), config.getTaskNum(), config.getDelimiter());
        reader.initialize(new FileSplit(testFile));

        // only for PAMA test data
//        reader.setLabelFromOne(false);

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, config.getBatchSize(), config.getLabelIndex(), config.getNumClasses());

        if (config.isNoraml()) {
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(iterator);
            iterator.setPreProcessor(normalizer);
        }

        Evaluation eval = network.evaluate(iterator);

        System.out.println(eval.stats(true));
    }

}
