package org.nd4j.examples.ZacTest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DataProcessNew {


    public static void main(String[] args) throws Exception {

        int channels = 9;
        int height = 1;
        int width = 128;
        int labelNum = 6;


        String train_path = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/Har/x_train.csv";
        INDArray lineMat = Nd4j.readNumpy(train_path, ",");

        long h = lineMat.shape()[0];

        INDArray features = Nd4j.createUninitialized(new long[]{h, channels, height, width}, 'c');
        /**
         * if the code is running on a GPU-based device:
         *
         * then host is for cpu store, device is for gpu store
         *
         * otherwise, will do nothing...
         *
         */
//        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST);
        long w = lineMat.shape()[1];
        INDArray l = lineMat.get(NDArrayIndex.all(), NDArrayIndex.interval(w - 1, w));
        INDArray f = lineMat.get(NDArrayIndex.all(), NDArrayIndex.interval(0, w - 1));


        for (int i = 0; i < h; i++) {
            INDArray row = f.getRow(i);
            // reshape features to split to multi-channel
            INDArray channelRow = row.reshape('c', new long[]{channels, height, width});
            features.get(NDArrayIndex.point(i)).assign(channelRow);
        }

        INDArray labels = Nd4j.create(h, labelNum, 'c');
//        int[] array = l.data().asInt();
        for (int i = 0; i < l.rows(); i++) {
//            int label = array[i];
            labels.putScalar(i, l.getInt(i) - 1, 1.0f);
        }

        System.out.println(labels);
    }
}
