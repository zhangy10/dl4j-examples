package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

public class WriteTest {


    public static void main(String[] args) throws Exception {


        File f = new File("/Users/zhangyu/Desktop/zac_test.csv");

        BufferedOutputStream fout = new BufferedOutputStream(new FileOutputStream(f));

        INDArray random = Nd4j.rand(1, 10);

        String split = ",";


        // not work for each single one
//        List<Writable> list = RecordConverter.toRecord(random);
        List<Writable> list = toList(random);

        // write...
//        CSVRecordWriter writer = new CSVRecordWriter();
//        Partitioner p = new NumberOfRecordsPartitioner();
//        writer.initialize(new FileSplit(f), p);
//
//        writer.write(list);
//        writer.close();

        for (int i = 0; i < list.size(); i++) {
            Writable w = list.get(i);
            System.out.println(w.toString());
            fout.write(w.toString().getBytes(CSVRecordWriter.DEFAULT_CHARSET));
            if (i == list.size() - 1) {
                // write a new line
                fout.write(FileRecordWriter.NEW_LINE.getBytes());
            } else {
                // write split
                fout.write(CSVRecordWriter.DEFAULT_DELIMITER.getBytes(CSVRecordWriter.DEFAULT_CHARSET));
            }
        }

        fout.flush();
        fout.close();

    }

    public static List<Writable> toList(INDArray line) {
        List<Writable> lineList = new ArrayList<>();
        for (int i = 0; i < line.columns(); i++) {
            INDArray item = line.get(NDArrayIndex.point(i));
            lineList.add(new NDArrayWritable(item));
        }
        return lineList;
    }
}
