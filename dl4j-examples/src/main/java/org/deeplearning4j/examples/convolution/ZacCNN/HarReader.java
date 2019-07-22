package org.deeplearning4j.examples.convolution.ZacCNN;

import org.datavec.api.records.Record;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HarReader extends CSVRecordReader {

    protected int channels = 0;
    protected int taskNum = 0;
    protected int height = 0;
    protected int width = 0;
    protected int labelNum = 0;
    protected boolean labelFromOne = true;
    private boolean isNext = true;
    private int taskLeft = 0;

    public HarReader(int height, int width, int channels, int labelNum, int taskNum, char delimiter) {
        this(0, height, width, channels, labelNum, taskNum, delimiter);
    }

    public HarReader(int skipnum, int height, int width, int channels, int labelNum, int taskNum, char delimiter) {
        super(skipnum, delimiter);
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.labelNum = labelNum;
        this.taskNum = taskNum;
    }

    public void setLabelFromOne(boolean fromOne) {
        this.labelFromOne = fromOne;
    }

    @Override
    public boolean hasNext() {
        return isNext && super.hasNext();
    }

    @Override
    public void reset() {
        taskLeft = taskNum;
        isNext = true;
        super.reset();
    }

    /**
     * will return one multi-channel data
     *
     * @return
     */
    @Override
    public List<Writable> next() {
        long t = System.currentTimeMillis();
        List<Writable> line = super.next();

//        System.out.println("Read one line time: " + (System.currentTimeMillis() - t));
        t = System.currentTimeMillis();

        // reshape one line to multi-channel data
        Writable l = line.remove(line.size() - 1);
        INDArray f = RecordConverter.toArray(line);
        INDArray cf = f.reshape('c', new long[]{channels, height, width});
        INDArray rcf = cf.reshape(ArrayUtil.combine(new long[]{1}, cf.shape()));
        List<Writable> ret = RecordConverter.toRecord(rcf);
        ret.add(l);

//        System.out.println("Reshape one line and split feature + label time: " + (System.currentTimeMillis() - t));
        return ret;
    }

    /**
     * by batch reading
     *
     * @param num batch size
     * @return Feature & Label by 0-1 for each batch
     */
    @Override
    public List<List<Writable>> next(int num) {
        // check task number left
        taskLeft -= num;
        if (taskLeft < 0) {
            // take the rest of task, then stop loop
            num = Math.abs(taskLeft);
        }
        /**
         * ret.get(0) -> feature by batch
         *
         * ret.get(1) -> binary label by batch
         */

        long t = System.currentTimeMillis();
//        System.out.println("\n");

        List<INDArray> ret = new ArrayList<>();
        List<List<Writable>> temp = load(num);

//        System.out.println("1. Read a batch time: " + (System.currentTimeMillis() - t));
        t = System.currentTimeMillis();

        int rows = temp.size();

        // Nd4j.createUninitialized will not initial value of array
        INDArray features = Nd4j.createUninitialized(new long[]{rows, channels, height, width}, 'c');
        /**
         * if the code is running on a GPU-based device:
         *
         * then host is for cpu store, device is for gpu store
         *
         * otherwise, will do nothing...
         *
         */
//        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST);
        INDArray lineMat = RecordConverter.toMatrix(temp);
        temp.clear();

//        System.out.println("2. RecordConverter from List to NDArray: " + (System.currentTimeMillis() - t));
        t = System.currentTimeMillis();

        long w = lineMat.shape()[1];
        INDArray l = lineMat.get(NDArrayIndex.all(), NDArrayIndex.interval(w - 1, w));
        INDArray f = lineMat.get(NDArrayIndex.all(), NDArrayIndex.interval(0, w - 1));


//        System.out.println("3. add to NDArray: " + (System.currentTimeMillis() - t));
        t = System.currentTimeMillis();

        long h = lineMat.shape()[0];
        for (int i = 0; i < h; i++) {
            INDArray row = f.getRow(i);
            // reshape features to split to multi-channel
            INDArray channelRow = row.reshape('c', new long[]{channels, height, width});
            features.get(NDArrayIndex.point(i)).assign(channelRow);
        }


//        System.out.println("4. Reshape: " + (System.currentTimeMillis() - t));
        t = System.currentTimeMillis();

        // convert label to binary matrix
        INDArray labels = Nd4j.create(rows, labelNum, 'c');
        int offset = labelFromOne ? 1 : 0;
        for (int i = 0; i < l.rows(); i++) {
            labels.putScalar(i, l.getInt(i) - offset, 1.0f);
        }

        // key for channel
        ret.add(features);
        ret.add(labels);

//        System.out.println("5. Label convert: " + (System.currentTimeMillis() - t));

        // if task number is done, then stop loop
        if (taskLeft <= 0) {
            isNext = false;
        }
        return new NDArrayRecordBatch(ret);
    }

    private List<List<Writable>> load(int num) {
        List<List<Writable>> ret = new ArrayList<>(Math.min(num, 10000));
        int recordsRead = 0;
        while (hasNext() && recordsRead++ < num) {
            ret.add(super.next());
        }
        return ret;
    }

    public int getChannels() {
        return channels;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
