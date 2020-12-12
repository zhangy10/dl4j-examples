package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base.BaseTrain;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.GradientInfo;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.SystemRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.Message;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

public class TrainMain extends BaseTrain {

    // Chain use
    private boolean isLinked = false;
    public boolean isEnd = false;

    private List<Double> scales = new ArrayList<>();
    private List<Double> changes = new ArrayList<>();

    public TrainMain(BlockingQueue<Message> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile) {
        super(sendQueue, id, settings, splitListener, modelFile);
    }

    public TrainMain(BlockingQueue<Message> sendQueue, int id, Config settings, boolean isLinked, SplitListener splitListener, String modelFile) {
        super(sendQueue, id, settings, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainMain(int id, Config settings, int slaveNum, boolean isLinked, SplitListener splitListener, String modelFile) {
        super(id, settings, slaveNum, splitListener, modelFile);
        this.isLinked = isLinked;
    }

    public TrainMain(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile) {
        super(id, settings, slaveNum, splitListener, modelFile);
    }

    private TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
            batchID = iteration;
            if (isMaster) {
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                long bend = System.currentTimeMillis();
                long time = bend - bstart;
                bstart = bend;
                System.out.println("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch +
                                       " learning rate: " + network.getLearningRate(0) + " time: " + time);

                batchTime.add(time);
            }

            // get average loss value
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());

            switch (SystemRun.policy) {
                case BATCH:
                    sync(model);
                    break;
//                case HALF_EPOC:
//                    break;
            }

            // add l1 changes
            appendGradient(epoc, model.params().norm1Number().doubleValue());
        }

        double l1start = 0;

        double l1end = 0;

        double l1gradient = 0;

        double lastChange = 0;

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();

            // get L1 start point
            l1start = model.params().norm1Number().doubleValue();
            appendGradient(epoc, l1start);
        }

        @Override
        public void onEpochEnd(Model model) {
            System.out.println("[-------------onEpochEnd----------------] batchID: " + batchID);

            // get L1 end point
            l1end = model.params().norm1Number().doubleValue();
            l1gradient = l1start - l1end;
            GradientInfo.append(id, l1start, l1end, l1gradient);

            if (SystemRun.policy == SystemRun.SyncPolicy.EPOC) {
                sync(model);
            }

            // update done, and test for accuracy
            if (SystemRun.isTestRound) {
                test((MultiLayerNetwork) model);
            }
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {

        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {

        }

        @Override
        public void onGradientCalculation(Model model) {

        }

        @Override
        public void onBackwardPass(Model model) {

        }

        private void sync(Model model) {
            System.out.println("[----------SYNC-----------] batchID: " + batchID);

            // average loss value
            List<Double> lossList = epocLoss.get(epoc);
            double loss = 0;
            for (Double d : lossList) {
                loss += d;
            }
            loss /= lossList.size();

            MultiLayerNetwork network = (MultiLayerNetwork) model;

            // each epoch end, then will do weight sync
            if (isMaster) {
                List<Message> msgList = new ArrayList<>();
                try {
                    int num = 0;
                    if (isLinked) {
                        if (avaliableNum == 0) {
                            num = 0;
                        } else {
                            // will only get 1 result from sub node
                            num = 1;
                        }
                    } else {
                        num = avaliableNum;
                    }
                    while (num > 0) {
                        System.out.println("Master is [waiting]... left: " + num);
                        msgList.add(getQueue.take());
                        num--;
                        System.out.println("Master is taking... left: " + num);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                INDArray newP = model.params().dup();

                if (isLinked && avaliableNum > 0) {
                    Message newMsg = msgList.get(0);

                    newMsg.num++;
                    newMsg.parameters.muli(newMsg.num - 1);
                    newP.addi(newMsg.parameters);
                    newP.divi(newMsg.num);
                    System.out.println("Master is divided by: [" + newMsg.num + "]");
                } else {
                    // 1.  average SGD
                    for (Message m : msgList) {
                        newP.addi(m.parameters);
                    }

                    // average, but learning rate will be smaller and smaller
                    newP.divi(msgList.size() + 1);

                    if (SystemRun.isScaleDecay) {
//                        && l1gradient > 0
                        // weight decay and get scale
//                        newP.muli(0.75);
                        newP.muli(getScale(l1start, l1end, msgList));
                    }
                }

                // update model
                // * setParam.assign will make a copy
                model.setParams(newP);

//                // update done, and test for accuracy
//                if (SystemRun.isTestRound) {
//                    test((MultiLayerNetwork) model);
//                }

                // fixed bug: if not send message back to slave, the memory will not be relesaed
                Message newMsg = new Message(id);
                newMsg.parameters = newP;
                int i = 0;
                for (BlockingQueue queue : broadcast) {
                    queue.offer(newMsg);
                    i++;
                    System.out.println("master sending to " + i);
                }
            } else {
                Message message = new Message(id);

                // if linked, need frist get message from sub node, then send to root
                if (isLinked && !isEnd) {
                    try {
                        System.out.println("node is waiting for sub node... thread: " + id);
                        Message newMsg = getQueue.take();
                        message.parameters = model.params().dup();

                        // add with sub node weights
                        newMsg.num++;
                        newMsg.parameters.muli(newMsg.num - 1);
                        message.parameters.addi(newMsg.parameters);
                        message.parameters.divi(newMsg.num);
                        message.num = newMsg.num;

                        System.out.println("divide by: [" + message.num + "] thread: " + id);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    message.parameters = model.params();
                    message.gradient = l1gradient;
                    message.num++;
                }

                message.id = id;

                System.out.println("Slave is sending... thread: " + id);
                masterQueue.offer(message);

                Message newMsg = null;
                try {
                    System.out.println("Slave is waiting for master... thread: " + id);
                    newMsg = getQueue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Slave get new P.... thread: " + id);

                INDArray newP = newMsg.parameters;

                model.setParams(newP);
            }
        }

        /**
         * for averaging scale
         *
         * @param start
         * @param end
         * @param list
         * @return
         */
        private double getScale(double start, double end, List<Message> list) {
            double base = end;
            double g = 0;
            for (Message m : list) {
                base += start - m.gradient;
                g += m.gradient;
            }
            base = base / (double) (list.size() + 1);
            g = g / (double) (list.size() + 1);
            double scale = 0;

            double otherG = list.size() * g;
//            if (g < 0) {
//                double l = base / start;
////                double l = 0.3;
//                otherG = otherG * l;
//            }

            // ver1
//            scale = Math.abs((base - otherG) / Math.max(base, otherG));

            // ver2
            double change = Math.abs(start - (list.size() + 1) * g);
            System.out.println("\n\n [[[[[-------------------> Change: " + change + ", LastChange: " + lastChange + " <--------------------]]]]] \n\n");
            if (SystemRun.isMobileNet) {
                lastChange = change;
                scale = Math.abs(change / Math.max(start, (list.size() + 1) * g));
            } else {
                // minimize difference
                if (lastChange == 0 || lastChange > change) {
                    lastChange = change;
                    scale = Math.abs(change / Math.max(start, (list.size() + 1) * g));
                } else {
                    // no impact
                    scale = 1;
                }
            }

            changes.add(lastChange);
            scales.add(scale);
            System.out.println("-----------------\n\nThe scale is: " + scale + " Epoch: " + epoc + "\n\n-----------------");
            return scale;
        }
    };

    @Override
    protected void loadData() {
        trainIterator = loading(settings.getDataPath(), settings.getTaskNum());
        testIterator = loading(settings.getTestPath(), SystemRun.isIID ? settings.getTaskNum() * (slaveNum + 1) : settings.getTaskNum());
    }

    @Override
    protected MultiLayerNetwork getModel() {
        MultiLayerNetwork model = null;
        MultiLayerConfiguration conf = null;
        if (isMaster) {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(existingFile);
            } catch (Exception e) {
                System.out.println("No existing file found, model will be initialzed in MAIN");
            } finally {
                if (model == null) {
                    conf = MDLModel.getNetwork(modelType, settings);
                    model = new MultiLayerNetwork(conf);
                    model.init();
                }
            }

            System.out.println(model.summary());

            // for test, save model
            try {
                Nd4j.saveBinary(model.params(), new File("/Users/zhangyu/Desktop/cache"));
            } catch (IOException e) {
                e.printStackTrace();
            }

            // send conf to others
            Message message = new Message(id);
            if (conf != null) {
                message.confJosn = conf.toJson();
            }
            // send init to others
            message.parameters = model.params();
            if (broadcast != null) {
                int i = 0;
                for (BlockingQueue send : broadcast) {
                    send.offer(message);
                    i++;
                    System.out.println("send model init to " + i);
                }
            }
        } else {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(existingFile);
            } catch (Exception e) {
                System.out.println("No existing file found, model will be initialzed: id = " + id);
            } finally {
                if (model == null) {
                    try {
                        // read from master
                        Message message = getQueue.take();
                        System.out.println("Thread " + id + " init model.....");
                        conf = MultiLayerConfiguration.fromJson(message.confJosn);
                        model = new MultiLayerNetwork(conf);

                        // not use copy from master
                        model.init();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        model.setListeners(listener);
        return model;
    }

    @Override
    protected void afterTrain(String output, Model model) {
        // train done
        if (isMaster) {
            // add changes
            String changeList = "Change = " + changes.toString() + "\n\n";
            output += changeList;
            System.out.println(changeList);

            // add scales
            String scaleList = "scale = " + scales.toString() + "\n\n";
            output += scaleList;
            System.out.println(scaleList);

            // L1 gradient print
            String gradients = printAllGradient();
            // gather all gradient info
            int num = avaliableNum;
            while (num > 0) {
                System.out.println("Master is [waiting]... left: " + num);
                try {
                    Message msg = getQueue.take();
                    gradients += msg.log;
                    output += msg.output;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                num--;
                System.out.println("Master is taking... left: " + num);
            }
            output += gradients;
            System.out.println(gradients);

            if (splitListener != null) {
                splitListener.trainDone(output);
            }
        } else {
            // send gradients to master
            Message msg = new Message(id);
            msg.log = printAllGradient();
            msg.output = output;
            System.out.println("Slave is [FINALY] sending... thread: " + id);
            masterQueue.offer(msg);
        }
    }
}

