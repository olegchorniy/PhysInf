package com.physinf.lab1.sequence;

import java.util.function.Function;

/**
 * Created by ochorny on 14.11.2017.
 */
public class LogisticBinarizer implements Function<Double, Boolean> {

    @Override
    public Boolean apply(Double value) {
        return value > 0.5;
    }
}
