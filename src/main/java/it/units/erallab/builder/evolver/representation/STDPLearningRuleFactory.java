package it.units.erallab.builder.evolver.representation;

import it.units.erallab.hmsrobots.core.controllers.snn.learning.*;
import it.units.malelab.jgea.core.IndependentFactory;

import java.util.Random;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

public class STDPLearningRuleFactory implements IndependentFactory<STDPLearningRule> {

  private final double pSymmetric;
  private final double pHebbian;

  private static final double[] MIN_SYMMETRIC_PARAMS = {1d, 1d, 3.5, 13.5};
  private static final double[] MAX_SYMMETRIC_PARAMS = {10.6, 44d, 10d, 20d};

  private static final double[] MIN_ASYMMETRIC_PARAMS = {0.1, 0.1, 1d, 1d};
  private static final double[] MAX_ASYMMETRIC_PARAMS = {1d, 1d, 10d, 10d};

  public STDPLearningRuleFactory(double pSymmetric, double pHebbian) {
    this.pSymmetric = pSymmetric;
    this.pHebbian = pHebbian;
  }

  public STDPLearningRuleFactory() {
    pSymmetric = 0.5;
    pHebbian = 0.5;
  }

  @Override
  public STDPLearningRule build(RandomGenerator random) {
    double pSymmetric = random.nextDouble();
    double pHebbian = random.nextDouble();
    STDPLearningRule learningRule;
    double[] params;
    if (pSymmetric <= this.pSymmetric) {
      params = IntStream.range(0, MIN_SYMMETRIC_PARAMS.length).mapToDouble(i -> random.nextDouble() * (MAX_SYMMETRIC_PARAMS[i] - MIN_SYMMETRIC_PARAMS[i]) + MIN_SYMMETRIC_PARAMS[i]).toArray();
      learningRule = (pHebbian <= this.pHebbian) ? new SymmetricHebbianLearningRule() : new SymmetricAntiHebbianLearningRule();
    } else {
      params = IntStream.range(0, MIN_ASYMMETRIC_PARAMS.length).mapToDouble(i -> random.nextDouble() * (MAX_ASYMMETRIC_PARAMS[i] - MIN_ASYMMETRIC_PARAMS[i]) + MIN_ASYMMETRIC_PARAMS[i]).toArray();
      learningRule = (pHebbian <= this.pHebbian) ? new AsymmetricHebbianLearningRule() : new AsymmetricAntiHebbianLearningRule();
    }
    learningRule.setParams(params);
    return learningRule;
  }
}
