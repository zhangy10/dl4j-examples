package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ZacMDP extends ZacAbsMDP {

    private Random random = new Random(1234);
    private List<Integer> vaildActions = new ArrayList<>();

    // current step
    protected ZacStep currentStep;
    // current reply
    protected StepReply currentReply;

    public ZacMDP(int actionSize, int busySize) {
        super(actionSize, busySize);
        currentStep = reset();
    }

    private void initVaildActions() {
        // renew actions
        for (int i = 0; i < ACTION_SIZE; i++) {
            vaildActions.add(i);
        }
    }

    public ZacStep getCurrentStep() {
        return currentStep;
    }

    public StepReply<ZacStep> getCurrentReply() {
        return currentReply;
    }

    @Override
    public ZacStep getNextAction() {
        // TODO get vaild next action
        int length = vaildActions.size();
        currentStep = new ZacStep(currentStep.getValues(), vaildActions.get(random.nextInt(length)));
        System.out.println("Select next action: " + currentStep.toString());
        return currentStep;
    }


    @Override
    public StepReply<ZacStep> step(Integer integer) {
        double reward = 0.0;
        // remove selected action from vaildActions
        vaildActions.remove(integer);



//        currentStep.setValues();
        currentReply = new StepReply(currentStep, reward, isDone(), new JSONObject("{}"));
        return currentReply;
    }

    @Override
    public ObservationSpace<ZacStep> getObservationSpace() {
        return observationSpace;
    }


    @Override
    public boolean isDone() {


        return false;
    }

    private boolean isFinish() {


        return false;
    }

    @Override
    public ZacStep reset() {
        initVaildActions();
        return super.reset();
    }

    @Override
    public MDP<ZacStep, Integer, DiscreteSpace> newInstance() {
        return new ZacMDP(ACTION_SIZE, BUSY_SIZE);
    }
}
