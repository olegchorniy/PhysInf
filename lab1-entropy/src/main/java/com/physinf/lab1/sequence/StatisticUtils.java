package com.physinf.lab1.sequence;

import one.util.streamex.EntryStream;

import java.util.Arrays;
import java.util.Map;
import java.util.SortedMap;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

public abstract class StatisticUtils {

    private static final Collector<Integer, ?, Map<Integer, Long>> frequenciesCounter =
            Collectors.groupingBy(Function.identity(), Collectors.counting());

    private StatisticUtils() {
    }

    public static SortedMap<Integer, Double> empiricDistribution(int[] values) {
        Map<Integer, Long> frequencies = Arrays.stream(values)
                .boxed()
                .collect(frequenciesCounter);

        return EntryStream.of(frequencies)
                .mapValues(counter -> counter / (double) values.length)
                .toSortedMap();
    }
}
