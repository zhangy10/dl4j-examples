package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.nd4j.linalg.schedule.*;

import java.util.ArrayList;
import java.util.List;

public class LearnRateTest {



    public static void main(String[] args) {

        int epoc = 20;

//        ExponentialSchedule is = new ExponentialSchedule(ScheduleType.EPOCH, 0.001, 0.99);


        InverseSchedule is = new InverseSchedule(ScheduleType.EPOCH, 0.02, 0.99, 1);


//        StepSchedule is = new StepSchedule(ScheduleType.EPOCH, 0.001, 0.99, 20);

//        StepSchedule is =  new StepSchedule(ScheduleType.EPOCH, 2e-2, 0.1, 100000);

//        SigmoidSchedule is = new SigmoidSchedule(ScheduleType.EPOCH, 0.001, 0.99, 20);


//        PolySchedule is = new PolySchedule(ScheduleType.EPOCH, 0.001, 1, 20);

        List<Double> list = new ArrayList<>();

        for (int i = 0; i < epoc; i++) {
            list.add(is.valueAt(0, i));
        }
        System.out.println(list);
    }
}
