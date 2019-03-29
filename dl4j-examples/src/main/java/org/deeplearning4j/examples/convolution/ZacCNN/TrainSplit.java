package org.deeplearning4j.examples.convolution.ZacCNN;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class TrainSplit extends Thread {

    private int id;

    private BlockingQueue<Msg> getQueue = new LinkedBlockingQueue<>();

    private BlockingQueue<Msg> sendQueue;

    private List<BlockingQueue<Msg>> broadcast;

    private boolean isMaster = false;

    private Settings settings;

    private int slaveNum;

    private int epoc;

    private Map<Integer, List<Double>> epocLoss = new TreeMap<>();


    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Settings settings) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;
    }

    public TrainSplit(int id, Settings settings, int slaveNum) {
        this(null, id, settings);
        this.isMaster = true;
        this.slaveNum = slaveNum;
        broadcast = new ArrayList<>();
    }

    public BlockingQueue getQueue() {
        return getQueue;
    }

    public void addSlave(BlockingQueue<Msg> queue) {
        if (broadcast != null) {
            broadcast.add(queue);
        }
    }


    private TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
            if (isMaster) {
                System.out.println("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch);
                List<Double> list = epocLoss.get(epoch);
                if (list == null) {
                    list = new ArrayList<>();
                    epocLoss.put(epoch, list);
                }
                list.add(model.score());
            }
        }

        @Override
        public void onEpochStart(Model model) {

        }

        @Override
        public void onEpochEnd(Model model) {
            // each epoch end, then will do weight sync
            if (isMaster) {
                if (slaveNum > 0) {
                    List<Msg> msgList = new ArrayList<>();
                    try {
                        int num = slaveNum;
                        while (num > 0) {
                            msgList.add(getQueue.take());
                            num--;
                            System.out.println("Master is taking... left: " + num);
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    // weight sync
                    System.out.println("SGD +++++++++");
                    INDArray newP = model.params();
                    for (Msg msg : msgList) {
                        newP = newP.add(msg.parameters);
                    }
                    newP = newP.div(msgList.size() + 1);
                    model.setParams(newP);

                    Msg newMsg = new Msg();
                    newMsg.parameters = newP;
                    int i = 0;
                    for (BlockingQueue queue : broadcast) {
                        queue.offer(newMsg);
                        i++;
                        System.out.println("master sending to " + i);
                    }
                }
            } else {
                Msg msg = new Msg();
                msg.parameters = model.params();
                System.out.println("Slave is sending... thread: " + id);
                sendQueue.offer(msg);

                Msg newMsg = null;
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
    };


    @Override
    public void run() {
        HarReader reader = new HarReader(settings.numLinesToSkip, settings.height,
            settings.width, settings.channel, settings.numClasses, settings.taskNum, settings.delimiter);
        try {
            reader.initialize(new FileSplit(settings.getFile()));
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, settings.batchSize, settings.labelIndex, settings.numClasses);

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        if (settings.isNoraml) {
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            iterator.setPreProcessor(normalizer);
        }

        MultiLayerNetwork model = null;
        if (isMaster) {
            MultiLayerConfiguration conf = getModelConf(settings.numClasses, settings.height, settings.width, settings.channel);
            // send conf to others
            Msg message = new Msg();
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();
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
                // read from master
                Msg message = getQueue.take();
                System.out.println("Thread " + id + " init model.....");
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(message.confJosn);
                model = new MultiLayerNetwork(conf);
                // clone will be false in real case
                model.init(message.parameters, true);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        model.setListeners(listener);
        model.fit(iterator, settings.epoch);

        // evaluate
        if (isMaster) {
            System.out.println("Save model....");
            String basePath = "/Users/zhangyu/Desktop/";
            try {
                ModelSerializer.writeModel(model, basePath + "multi_model.bin", true);
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("Save model done!!");

            // each epoc average
            List<Double> averageList = new ArrayList<>();
            Iterator<Integer> it = epocLoss.keySet().iterator();
            while (it.hasNext()) {
                int index = it.next();
                List<Double> epocList = epocLoss.get(index);
                int size = epocList.size();
                double average = 0;
                for (int i = 0; i < size; i++) {
                    average += epocList.get(i);
                }
                averageList.add(average / size);
            }

            System.out.println(averageList);
        }
    }

    private ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        //卷积输入层，参数包括名字，过滤器数量，输出节点数，卷积核大小，步副大小，补充边框大小，偏差
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private DenseLayer full(String name, int out, double bias, double dropOut) {
        //全连接层，本例中输出4096个节点，偏差为1，随机丢弃比例50%，参数服从均值为0，方差为0.005的高斯分布
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }


    public MultiLayerConfiguration getModelConf(int numLabels, int height, int width, int channels) {
        // ??
        double nonZeroBias = 1; //偏差
        double dropOut = 0.8; //随机丢弃比例
        long seed = 42;

        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
//                                           .weightInit(WeightInit.NORMAL) //根据给定的分布采样参数
                   .weightInit(WeightInit.DISTRIBUTION)
                   .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                   .activation(Activation.RELU)
                   .updater(new Adam(0.001))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))

                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                   .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   // update padding issue
//                                           .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
                   .layer(1, maxpooling("m1", new int[]{1, 2}, new int[]{1, 2}))
                   .layer(2, convNet("c2", -1, 72, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
//                                           .layer(2, convNet("c2", -1, 72, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 16}, nonZeroBias))
                   .layer(3, maxpooling("m2", new int[]{1, 2}, new int[]{1, 2}))
                   .layer(4, full("f1", 300, nonZeroBias, dropOut))
                   .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                 .name("o1")
                                 .nOut(numLabels)
                                 .activation(Activation.SOFTMAX)
                                 .build())
                   .backprop(true)
                   .setInputType(InputType.convolutional(height, width, channels))
                   .build();
    }

    public static class Settings {
        int epoch = 20;

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        int taskNum = 7352;

        char delimiter = ',';

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        // last pos is label
        // by channel, the label index will be the second one, not actual label index
        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row

        int numClasses = 6;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 16;

        // channel * width = inputwidth
        int channel = 9;
        int height = 1;
        int width = 128;

        boolean isNoraml = false;
        MultiLayerConfiguration conf;

        public Settings(int numLinesToSkip, int taskNum) {
            this.numLinesToSkip = numLinesToSkip;
            this.taskNum = taskNum;
        }

        public File getFile() throws Exception {
//            return new ClassPathResource("iris_test.txt").getFile();
            return new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_train.csv");
        }
    }

    public class Msg {
        INDArray parameters;
        String confJosn;
    }

    public static class Pair {
        int start;
        int end;

        public Pair(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "[" + start + "/" + end + "]";
        }
    }

    public static void main(String[] args) {
        int taskNum = 2;
        int total = 7352;
        List<Pair> list = getTask(taskNum, total);

        System.out.println(list.toString());

        TrainSplit master = null;
        List<TrainSplit> slaveList = new ArrayList<>();
        for (int i = 0; i < taskNum; i++) {
            Pair fragment = list.get(i);
            if (i == 0) {
                master = new TrainSplit(i, new Settings(fragment.start, fragment.end), taskNum - 1);
            } else {
                TrainSplit slave = new TrainSplit(master.getQueue(), i, new Settings(fragment.start, fragment.end));
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }
        master.start();
        for (TrainSplit t : slaveList) {
            t.start();
        }
    }

    public static List<Pair> getTask(int taskNum, int total) {
        int doNum = total / taskNum;
        int rest = total % taskNum;
        int lastNum = doNum + rest;

        List<Pair> list = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < taskNum; i++) {
            if (i == taskNum - 1) {
                list.add(new Pair(start, lastNum));
            } else {
                list.add(new Pair(start, doNum));
            }
            start += doNum;
        }
        return list;
    }
}
