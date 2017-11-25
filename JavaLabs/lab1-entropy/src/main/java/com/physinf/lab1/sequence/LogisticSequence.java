package com.physinf.lab1.sequence;

import java.util.Random;
import java.util.function.DoubleSupplier;

/**
 * x_n+1 = µ * x_n * (1 − x_n).
 */
public class LogisticSequence implements DoubleSupplier {

    private final double mu;
    private final double initialValue;

    private boolean initialReturned = false;
    private double prev;

    public LogisticSequence(double mu) {
        this(mu, new Random().nextDouble());
    }

    public LogisticSequence(double mu, double initialValue) {
        this.mu = mu;
        this.initialValue = initialValue;
    }

    @Override
    public double getAsDouble() {
        double nextValue;

        if (!this.initialReturned) {
            this.initialReturned = true;
            nextValue = this.initialValue;
        } else {
            nextValue = this.mu * this.prev * (1 - this.prev);
        }

        this.prev = nextValue;

        return nextValue;
    }
}
