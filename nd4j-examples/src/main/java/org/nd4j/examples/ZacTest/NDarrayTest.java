package org.nd4j.examples.ZacTest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NDarrayTest {


    public static void main(String[] args) {


        INDArray newP = Nd4j.rand(1, 4);
        INDArray mainP = Nd4j.rand(1, 4);
        INDArray newP1 = Nd4j.rand(1, 4);
        System.out.println("main: " + mainP);



        System.out.println("1: ");
        INDArray tmp  = mainP.dup();
        System.out.println(tmp);
        tmp.addi(newP);
        System.out.println(tmp);
        tmp.addi(newP1);
        System.out.println(tmp);
        System.out.println("main: " + mainP);


        System.out.println("2: ");
        INDArray tmpp  = mainP;
        tmpp = tmpp.add(newP);
        System.out.println(tmpp);
        tmpp = tmpp.add(newP1);
        System.out.println(tmpp);
        System.out.println("main: " + mainP);




//        System.out.println("newP" + newP);
//        System.out.println(mainP);
//
//        mainP.assign(newP);
//        System.out.println("assign: " + mainP);
//
//        newP.putScalar(0, -1);
//
//        System.out.println("newP" + newP);
//        System.out.println(mainP);

    }
}
