package org.nd4j.examples.ZacTest;

import org.nd4j.examples.numpy_cheatsheat.NumpyCheatSheat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DataProcess {

    private static List list = new ArrayList();

    static {
        list.add("body_acc_x_train.csv");
        list.add("body_acc_y_train.csv");
        list.add("body_acc_z_train.csv");
        list.add("body_gyro_x_train.csv");
        list.add("body_gyro_y_train.csv");
        list.add("body_gyro_z_train.csv");
        list.add("total_acc_x_train.csv");
        list.add("total_acc_y_train.csv");
        list.add("total_acc_z_train.csv");
    }

    public static void main(String[] args) {

        // train
        String train_path = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/New_data/Har/train";

        // train_y: y_train.txt
        String y_train = String.format("%s/%s", train_path, "y_train.csv");

        // test
        String test_path = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/New_data/Har/test";

        // test_y: y_test.txt
        String y_test = String.format("%s/%s", test_path, "y_test.csv");

        INDArray train_x = null;

        try {

            for (int i = 0; i < list.size(); i++) {
                String path = String.format("%s/%s", train_path, list.get(i));
                INDArray tmp = Nd4j.readNumpy(path, ",");
                if (train_x == null) {
                    train_x = Nd4j.zeros(tmp.shape()[0], 1);
                }
                train_x = Nd4j.hstack(train_x, tmp);
            }

            train_x = train_x.get(NDArrayIndex.all(), NDArrayIndex.interval(1, train_x.shape()[1]));

            INDArray X = null;
//            for (int i = 0; i < train_x.shape()[0]; i++) {
//                INDArray row = train_x.get(NDArrayIndex.point(i), NDArrayIndex.all());
//                row = row.reshape(new int[]{9, 128}).transpose();
//                if (X == null) {
//                    X = Nd4j.zeros(train_x.shape()[0], 128, 9);
//                }
//                X.get(NDArrayIndex.point(i)).assign(row);
//            }


            INDArray features = Nd4j.createUninitialized(new long[]{train_x.shape()[0], 9, 1, 128}, 'c');
            for (int i = 0; i < train_x.shape()[0]; i++) {
                INDArray row = train_x.get(NDArrayIndex.point(i), NDArrayIndex.all());

                long length = row.shape()[1];
                INDArray f = row.get(NDArrayIndex.interval(0, length - 2));
                INDArray l = row.get(NDArrayIndex.point(length - 1));

                row = f.reshape('c', new long[]{9, 1, 128});
                features.get(NDArrayIndex.point(i)).assign(row);

            }

            INDArray Y = Nd4j.readNumpy(y_train, ",");


//            File f = new File("", "zber");
////
////            Nd4j.saveBinary(X, f);

            System.out.println(String.format("success! x shape %l, %l; y shape %l, %l", X.shape()[0], X.shape()[1], Y.shape()[0], Y.shape()[1]));

//            INDArray T = Nd4j.readBinary(f);

//            System.out.println(T == null ? "no " : T.shape().toString());
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
