package com.physinf.lab1.sequence;

import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.stream.DoubleStream;

public abstract class SeqUtils {

    private SeqUtils() {
    }

    public static double[] generate(DoubleSupplier supplier, int n) {
        return DoubleStream.generate(supplier)
                .limit(n)
                .toArray();
    }

    public static int[] binarize(double[] values, Function<Double, Boolean> binarizer) {
        int[] binaries = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            binaries[i] = binarizer.apply(values[i]) ? 1 : 0;
        }

        return binaries;
    }

    public static int[] makeBlocks(int[] binaryValues, int blockLen) {
        int blockNum = binaryValues.length / blockLen;
        int[] blocks = new int[blockNum];

        for (int i = 0; i < blockNum; i++) {
            blocks[i] = makeBlock(binaryValues, i * blockLen, blockLen);
        }

        return blocks;
    }

    public static int makeBlock(int[] binaryValues, int fromIndex, int blockLen) {
        int block = 0;

        for (int i = 0; i < blockLen; i++) {
            block |= ((binaryValues[fromIndex + i] & 1) << i);
        }

        return block;
    }
}
