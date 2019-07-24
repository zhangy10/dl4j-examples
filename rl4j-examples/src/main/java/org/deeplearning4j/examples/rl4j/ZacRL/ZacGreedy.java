package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;

import java.util.Random;

public class ZacGreedy extends EpsGreedy {


    public ZacGreedy(Policy policy, MDP mdp, int updateStart, int epsilonNbStep, Random rd, float minEpsilon, StepCountable learning) {
        super(policy, mdp, updateStart, epsilonNbStep, rd, minEpsilon, learning);
    }


    @Override
    public float getEpsilon() {

        /**
         *
         * //TODO
         *
         *
         */
        return super.getEpsilon();
    }
}
