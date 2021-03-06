package org.deeplearning4j.examples.rl4j.RL;

import org.deeplearning4j.rl4j.mdp.toy.HardToyState;
import org.deeplearning4j.rl4j.space.Encodable;

import java.util.Arrays;

public class ZacStep implements Encodable{
    private final double[] values;
    private final int step;

    @Override
    public double[] toArray() {
        return this.values;
    }

    public ZacStep(double[] values, int step) {
        this.values = values;
        this.step = step;
    }

    public double[] getValues() {
        return this.values;
    }

    public int getStep() {
        return this.step;
    }

    public void setValues(int index,double value) {
        values[index] = value;
    }

    public boolean equals(Object o) {
        if (o == this) {
            return true;
        } else if (!(o instanceof HardToyState)) {
            return false;
        } else {
            HardToyState other = (HardToyState)o;
            if (!Arrays.equals(this.getValues(), other.getValues())) {
                return false;
            } else {
                return this.getStep() == other.getStep();
            }
        }
    }

//    public int hashCode() {
//        int PRIME = true;
//        int result = 1;
//        int result = result * 59 + Arrays.hashCode(this.getValues());
//        result = result * 59 + this.getAction();
//        return result;
//    }

    public String toString() {
        return "ZacStep: (values=" + Arrays.toString(this.getValues()) + ", step=" + this.getStep() + ")";
    }

    public ZacStep clone() {
        double[] newValues = new double[values.length];
        System.arraycopy(values, 0, newValues, 0, values.length);
        return new ZacStep(newValues, step);
    }
}

