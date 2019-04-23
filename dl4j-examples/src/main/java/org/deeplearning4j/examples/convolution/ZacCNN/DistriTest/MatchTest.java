package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MatchTest {


    /**
     *      * 正则表达式匹配两个指定字符串中间的内容
     *      * @param soap
     *      * @return
     *     
     */
    public static List<Poistion> getSubUtil(String soap, String rgex) {
        List<Poistion> list = new ArrayList<Poistion>();
        Pattern pattern = Pattern.compile(rgex);// 匹配的模式
        Matcher m = pattern.matcher(soap);
        while (m.find()) {
            int i = 1;
            Poistion p = new Poistion();
            p.start = m.start(i);
            p.end = m.end(i);
            p.text = m.group(i);
            list.add(p);
            i++;
        }
        return list;
    }

    /**
     *      * 返回单个字符串，若匹配到多个的话就返回第一个，方法与getSubUtil一样
     *      * @param soap
     *      * @param rgex
     *      * @return
     *     
     */
    public static Poistion getSubUtilSimple(String soap, String rgex) {
        Pattern pattern = Pattern.compile(rgex);// 匹配的模式
        Matcher m = pattern.matcher(soap);
        while (m.find()) {
            Poistion p = new Poistion();
            p.start = m.start(1);
            p.end = m.end(1);
            p.text = m.group(1);
            return p;
        }
        return null;
    }

    static class Poistion {
        int start;
        int end;
        String text;

        @Override
        public String toString() {
            return start + " " + end + " " + text;
        }
    }

    /**
     *      * 测试
     *      * @param args
     *     
     */
    public static void main(String[] args) {
        String str = "!3443fgjhggf!!j";
        String rgex = "!(.*?)!";
        System.out.println(getSubUtilSimple(str, rgex));

        List<Poistion> list = getSubUtil(str, rgex);
        for (Poistion p : list) {
            System.out.println(p);
        }

    }


}
