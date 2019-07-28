package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

import java.util.*;

public abstract class ZacAbsMDP implements MDP<ZacStep, Integer, DiscreteSpace> {

    // for random initalize each step's values
    final protected static int SEED = 1234;

    // stationary env
    protected static int ACTION_SIZE = 5; // number of nodes
    protected static int BUSY_SIZE = 1;

    // enviornment values
    protected double[] Env;
    // Q learning model reference
    protected ZacQLearning currentDQN;
    protected static Map<StateKey, State> valueInitMap = new HashMap<>();
    protected static Map<State, Double> stateInitMap = new HashMap<>();

    protected DiscreteSpace actionSpace;
    protected ObservationSpace<ZacStep> observationSpace;

    // reward = r - decay - decayRate * currentStep
    public static double decay = 0.04;
    public static double decayRate = 0.1;

    static {
        stateInitMap.put(State.FREE, 0.9);
        stateInitMap.put(State.BUSY, 0.6);
        stateInitMap.put(State.DONE, 0.0);
        stateInitMap.put(State.SYNC_GET, 0.3);
        stateInitMap.put(State.SYNC_SEND, 0.4);

        valueInitMap.put(new StateKey(stateInitMap.get(State.FREE)), State.FREE);
        valueInitMap.put(new StateKey(stateInitMap.get(State.BUSY)), State.BUSY);
        valueInitMap.put(new StateKey(stateInitMap.get(State.DONE)), State.DONE);
        valueInitMap.put(new StateKey(stateInitMap.get(State.SYNC_GET)), State.SYNC_GET);
        valueInitMap.put(new StateKey(stateInitMap.get(State.SYNC_SEND)), State.SYNC_SEND);
    }

    public ZacAbsMDP(int actionSize, int busySize) {
        this.ACTION_SIZE = actionSize;
        this.BUSY_SIZE = busySize;
        this.actionSpace = new DiscreteSpace(ACTION_SIZE);
        this.observationSpace = new ArrayObservationSpace(new int[]{ACTION_SIZE});
        initEnv();
    }

    private double[] initEnv() {
        Env = new double[ACTION_SIZE];
        List<Integer> buysPos = getBusy();
        for (int j = 0; j < ACTION_SIZE; j++) {
            // TODO fix pos
            if (buysPos.contains(j)) {
                Env[j] = stateInitMap.get(State.BUSY);
            } else {
                Env[j] = stateInitMap.get(State.FREE);
            }
        }
        return Env;
    }

    private List<Integer> getBusy() {
        // random busy position
        Random r = new Random(SEED);
        List<Integer> busyList = new ArrayList<>();
        int left = BUSY_SIZE;
        while (left > 0) {
            int pos = r.nextInt(ACTION_SIZE);
            if (!busyList.contains(pos)) {
                busyList.add(pos);
                left--;
            }
        }
        return busyList;
    }

    public int getActionSize() {
        return ACTION_SIZE;
    }

    public double[] getEnvData() {
        return Env;
    }

    public void setDQN(QLearning dqn) {
        this.currentDQN = (ZacQLearning) dqn;
    }

    public QLearning getDQN() {
        return currentDQN;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public ZacStep reset() {
        return new ZacStep(getEnvData(), 0);
    }

    public abstract ZacStep getNextAction();

    @Override
    public void close() {
        System.out.println("close.......");
    }

    public enum State {
        FREE,
        BUSY,
        DONE,
        SYNC_SEND,
        SYNC_GET,
    }

    public static class StateKey {

        public double value;
        public int count = 0;

        public double getValue() {
            return value + count * decay;
        }

        public StateKey(double value) {
            this(value, 0);
        }

        public StateKey(double value, int count) {
            this.value = value;
            this.count = count;
        }

    }
}
