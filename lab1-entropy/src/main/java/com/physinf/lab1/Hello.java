package com.physinf.lab1;

import com.physinf.lab1.sequence.LogisticBinarizer;
import com.physinf.lab1.sequence.LogisticSequence;
import com.physinf.lab1.sequence.SeqUtils;
import com.physinf.lab1.sequence.StatisticUtils;

import java.util.SortedMap;

public class Hello {

    public static void main(String[] args) {

        for (double mu = 0.5; mu <= 4.2; mu += 0.1) {

            double[] values = SeqUtils.generate(new LogisticSequence(mu, 0.7692499279378784), 100_000);
            int[] binarized = SeqUtils.binarize(values, new LogisticBinarizer());

            int[] blocks = SeqUtils.makeBlocks(binarized, 1);

        /*System.out.println(Arrays.toString(values));
        System.out.println(Arrays.toString(binarized));*/

            System.out.println("mu = " + mu);

            SortedMap<Integer, Double> distribution = StatisticUtils.empiricDistribution(blocks);
            distribution.forEach((block, prob) -> {
                System.out.println(Integer.toBinaryString(block) + " : " + prob);
            });
            System.out.println();
        }
    }
}
