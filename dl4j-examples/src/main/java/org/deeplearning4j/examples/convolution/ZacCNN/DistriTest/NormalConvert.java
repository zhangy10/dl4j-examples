package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

public class NormalConvert {


    public static void main(String[] args) throws Exception {

        // train not normal
//        File trainFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/x_train.csv");
//        File trainFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/nor_shuffle_data/1_train.csv");
//        int trainNum = 7352;

//        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/nor_shuffle_data/1_test.csv");
        File testFile = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/test.csv");
        int trainNum = 9999;

        HarReader reader = new HarReader(0, 1, 128, 9, 6, trainNum, ',');
        reader.initialize(new FileSplit(testFile));

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, 16, 1, 6);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);


        // save to csv
        String split = ",";
//        File nor_train = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_train.csv");

        File nor_test = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_test.csv");
        BufferedOutputStream fout = new BufferedOutputStream(new FileOutputStream(nor_test));


//        DataOutputStream dout = new DataOutputStream()
        int count = 0;
        while (iterator.hasNext()) {
            // already prepocess.... did!!
            DataSet batch = iterator.next();

            // transform multi-channel to one line
            INDArray f = batch.getFeatures();
            INDArray l = batch.getLabels();
            long batchSize = f.shape()[0];
            for (int i = 0; i < batchSize; i++) {
                INDArray line = f.get(NDArrayIndex.point(i));
                line = line.reshape('c', new long[]{reader.getHeight(), reader.getChannels() * reader.getWidth()});
                INDArray labels = l.get(NDArrayIndex.point(i));

                INDArray la = Nd4j.zeros(1);
                for (int j = 0; j < labels.columns(); j++) {
                    if (labels.getInt(j) != 0) {
                        la.putScalar(0, j + 1);
                        break;
                    }
                }
                line = Nd4j.hstack(line, la);
                // write one line to csv
                List<Double> lineList = toList(line);
                for (int k = 0; k < lineList.size(); k++) {
                    Double w = lineList.get(k);
                    fout.write(w.toString().getBytes(CSVRecordWriter.DEFAULT_CHARSET));
                    if (k == lineList.size() - 1) {
                        // write a new line
                        fout.write(FileRecordWriter.NEW_LINE.getBytes());
                    } else {
                        // write split
                        fout.write(CSVRecordWriter.DEFAULT_DELIMITER.getBytes(CSVRecordWriter.DEFAULT_CHARSET));
                    }
                }
                count++;
                System.out.println("done: " + count);
            }
        }
        fout.flush();
        fout.close();
    }

    /**
     * can be able to keep full double value
     *
     * @param line
     * @return
     */
    public static List<Double> toList(INDArray line) {
        List<Double> lineList = new ArrayList<>();
        for (int i = 0; i < line.columns(); i++) {
            Double item = line.data().getDouble(i);
            lineList.add(item);
        }
        return lineList;
    }
}
