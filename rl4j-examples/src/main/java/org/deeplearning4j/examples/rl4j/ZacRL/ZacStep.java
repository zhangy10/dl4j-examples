package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.rl4j.mdp.toy.HardToyState;
import org.deeplearning4j.rl4j.space.Encodable;

import java.util.Arrays;

public class ZacStep implements Encodable {
    private double[] values;
    private int action;

    @Override
    public double[] toArray() {
        return this.values;
    }

    public ZacStep(double[] values, int step) {
        this.values = values;
        this.action = step;
    }

    public double[] getValues() {
        return this.values;
    }

    public int getAction() {
        return this.action;
    }

    public void setValues(double[] values) {
        this.values = values;
    }

    public String toString() {
        return "ZacStep: (values=" + Arrays.toString(this.getValues()) + ", action=" + this.getAction() + ")";
    }

    public ZacStep clone() {
        double[] newValues = new double[values.length];
        System.arraycopy(values, 0, newValues, 0, values.length);
        return new ZacStep(newValues, action);
    }
}

