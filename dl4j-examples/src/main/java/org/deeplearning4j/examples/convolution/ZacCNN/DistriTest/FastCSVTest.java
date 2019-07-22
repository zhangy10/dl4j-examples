package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import de.siegmar.fastcsv.reader.CsvParser;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRow;
import org.apache.commons.io.IOUtils;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.examples.convolution.ZacCNN.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.DataType;
import org.deeplearning4j.examples.convolution.ZacCNN.HarReader;

import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Iterator;

public class FastCSVTest {


    public static void main(String[] args) throws Exception {

//        fast1();
        // faster than 1
//        dl4jReader();
        System.out.println(readData());
    }


    public static void fast1() throws Exception {
        Config config = DataSet.getConfig(DataType.HAR);

        File file = new File(config.getDataPath());
        CsvReader csvReader = new CsvReader();


        CsvParser csvParser = csvReader.parse(file, StandardCharsets.UTF_8);
        CsvRow row;
        long start = System.currentTimeMillis();
        while ((row = csvParser.nextRow()) != null) {
            System.out.println("Read line: " + row);
//            System.out.println("First column of line: " + row.getField(0));
        }
        System.out.println(System.currentTimeMillis() - start);
    }


    public static void dl4jReader() throws Exception {

        Config config = DataSet.getConfig(DataType.HAR);

        Iterator<String> iter = IOUtils.lineIterator(new InputStreamReader(Files.newInputStream(
            new File(config.getDataPath()).toPath(), StandardOpenOption.READ)));

        long start = System.currentTimeMillis();
        while (iter.hasNext()) {
            System.out.println(iter.next());
        }
        System.out.println(System.currentTimeMillis() - start);
    }

    public static String readData() throws Exception{
        // test har read csv
        long start = System.currentTimeMillis();

        Config config = DataSet.getConfig(DataType.HAR);
        HarReader reader = new HarReader(config.getNumLinesToSkip(), config.getHeight(), config.getWidth(), config.getChannel(),
            config.getNumClasses(), config.getTaskNum(), config.getDelimiter());
        reader.initialize(new FileSplit(config.getFile()));

        // 50ms per line
        int index = 0;
//        while (reader.hasNext()) {
//            reader.next();
//            log.d("Read csv single data: " + ++index);
//        }
//
        long end = System.currentTimeMillis();
        long t1 = end - start;

        reader.reset();
        index = 0;
        // 1s per batch
        while (reader.hasNext()) {
            reader.next(config.getBatchSize());
//            System.out.println("Read csv batch data: " + ++index);
        }

        long t2 = System.currentTimeMillis() - end;
        return "single time: " + t1 + ", batch time: " + t2;

    }
}
