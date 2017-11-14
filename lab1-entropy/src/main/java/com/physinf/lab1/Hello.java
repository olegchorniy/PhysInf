package com.physinf.lab1;

import com.physinf.lab1.sequence.LogisticBinarizer;
import com.physinf.lab1.sequence.LogisticSequence;
import com.physinf.lab1.sequence.SeqUtils;
import com.physinf.lab1.sequence.StatisticUtils;

import java.util.Map;

public class Hello {

    // P that gives uniform distribution over binarized values
    private static final double MU_UNIFORM = 3.846;

    private static final double MU = 3.8;
    private static final int N = 10_000;
    private static final int MIN_LEN = 2;
    private static final int MAX_LEN = 10;

    public static void main(String[] args) {

        double[] values = SeqUtils.generate(new LogisticSequence(MU), N);
        int[] binarized = SeqUtils.binarize(values, new LogisticBinarizer());

        for (int blockLen = MIN_LEN; blockLen <= MAX_LEN; blockLen++) {
            int[] blocks = SeqUtils.makeBlocks(binarized, blockLen);

            Map<Integer, Double> distribution = StatisticUtils.empiricDistribution(blocks);

            double shannonEntropy = StatisticUtils.shannonEntropy(distribution);
            double renyiEntropy2 = StatisticUtils.renyiEntropy(distribution, 2.0);
            double renyiEntropy3 = StatisticUtils.renyiEntropy(distribution, 3.0);

            System.out.printf("%d\t%f\t%f\t%f%n", blockLen, shannonEntropy, renyiEntropy2, renyiEntropy3);
        }
    }

    private static void pOfMuTable() {
        for (double mu = 3.5; mu <= 4.0; mu += 0.001) {

            double[] values = SeqUtils.generate(new LogisticSequence(mu), 100_000);
            int[] binarized = SeqUtils.binarize(values, new LogisticBinarizer());
            int[] blocks = SeqUtils.makeBlocks(binarized, 1);

            Map<Integer, Double> distribution = StatisticUtils.empiricDistribution(blocks);

            System.out.printf("%.4f\t%.7f%n", mu, distribution.getOrDefault(1, 0.0));
        }
    }
}
