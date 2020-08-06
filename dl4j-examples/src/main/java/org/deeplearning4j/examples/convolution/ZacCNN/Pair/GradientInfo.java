package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import java.util.*;

public class GradientInfo extends ArrayList {

    public List<Double> start;
    public List<Double> end;
    public List<Double> gradient;

    GradientInfo() {
        start = new ArrayList<>();
        end = new ArrayList<>();
        gradient = new ArrayList<>();
        add(start);
        add(end);
        add(gradient);
    }

    private static Map<Integer, GradientInfo> gradientW = new HashMap<>();
    private static Map<Integer, String> name = new HashMap<>();

    static {
        name.put(0, "start");
        name.put(1, "end");
        name.put(2, "gard");
    }

    public synchronized static String printGradient() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n\n");
        synchronized (gradientW) {
            Set<Map.Entry<Integer, GradientInfo>> en = gradientW.entrySet();
            for (Map.Entry<Integer, GradientInfo> line : en) {
                Integer id = line.getKey();
                GradientInfo grad = line.getValue();
                // 3 lines
                for (int i = 0; i < grad.size(); i++) {
                    sb.append("T" + id + "_" + name.get(i) + " = [");
                    List<Double> gradData = (List<Double>) grad.get(i);
                    for (int j = 0; j < gradData.size(); j++) {
                        if (j == gradData.size() - 1) {
                            sb.append(gradData.get(j).toString() + "];\n");
                        } else {
                            sb.append(gradData.get(j).toString() + ",");
                        }
                    }
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    public synchronized static void append(int id, double start, double end, double gradient) {
        synchronized (gradientW) {
            GradientInfo gradientInfo = gradientW.get(id);
            if (gradientInfo == null) {
                gradientInfo = new GradientInfo();
                gradientW.put(id, gradientInfo);
            }
            gradientInfo.start.add(start);
            gradientInfo.end.add(end);
            gradientInfo.gradient.add(gradient);
        }
    }

}
