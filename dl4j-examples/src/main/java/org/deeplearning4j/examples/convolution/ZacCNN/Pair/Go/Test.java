package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import java.io.IOException;
import java.util.Random;

public class Test {

    static Random r = new Random();

    public static void main(String[] args) throws IOException {


//        Model network = ModelSerializer.restoreMultiLayerNetwork("/Users/zhangyu/Desktop/all.bin");
//
//
//        DNNModel model = (DNNModel) network;
//        System.out.println(model.summary());

//        int task = 6;
//
//        for (int i = 0; i < 30; i++) {
//            System.out.println(getRandomTarget(5));
//        }

        int size = 9;
        Random r1 = new Random();
        for (int i = 0; i < 30; i++) {
            System.out.print(r1.nextInt(size) + " ");
        }
        System.out.println();

        Random r2 = new Random(42);
        for (int i = 0; i < 30; i++) {
            System.out.print(r2.nextInt(size) + " ");
        }
        System.out.println();

        Random r3 = new Random(42);
        for (int i = 0; i < 30; i++) {
            System.out.print(r3.nextInt(size) + " ");
        }
        System.out.println();

        Random r4 = new Random();
        for (int i = 0; i < 30; i++) {
            System.out.print(r4.nextInt(size) + " ");
        }
        System.out.println();



//        INDArray shape = Nd4j.create(1, 1);
//        shape.putScalar(0, 0, 1.0);
//        System.out.println(shape);
//
//        INDArray out = Nd4j.create(1, 1);
//
//        double p = 0.01;
//
////        RandomBernoulli rb = new RandomBernoulli(shape, out, p);
//        int count = 0;
//        for (int i = 0; i < 10000; i++) {
//            boolean result = StdRandom.bernoulli(p);
//            if (result) {
//                count++;
//            }
//            System.out.println(result);
//        }
//        System.out.println(count);


//        Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(bernoullis.shape()), bernoullis), Nd4j.getRandom());

    }

    public static int getRandomTarget(int request) {
        TrainGo target = null;
        int sendTo = -1;
        while (true) {
            sendTo = r.nextInt(6);
            if (sendTo != request) {
                break;
            }
        }
        return sendTo;
    }
}
