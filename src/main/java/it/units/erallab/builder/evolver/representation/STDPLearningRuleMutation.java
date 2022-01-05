package it.units.erallab.builder.evolver.representation;

import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.SymmetricSTPDLearningRule;
import it.units.malelab.jgea.core.operator.Mutation;

import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

public class STDPLearningRuleMutation implements Mutation<STDPLearningRule> {

  private final double pChangeRule;
  private final STDPLearningRuleFactory stdpLearningRuleFactory;

  private static final double[] SYMMETRIC_PARAMS_STD = {1.92, 8.6, 1.3, 1.3};
  private static final double[] ASYMMETRIC_PARAMS_STD = {0.18, 0.18, 1.3, 1.3};

  public STDPLearningRuleMutation() {
    this(0.5, new STDPLearningRuleFactory());
  }

  public STDPLearningRuleMutation(double pChangeRule, STDPLearningRuleFactory stdpLearningRuleFactory) {
    this.pChangeRule = pChangeRule;
    this.stdpLearningRuleFactory = stdpLearningRuleFactory;
  }

  @Override
  public STDPLearningRule mutate(STDPLearningRule stdpLearningRule, RandomGenerator random) {
    if (random.nextDouble() <= pChangeRule) {
      return stdpLearningRuleFactory.build(random);
    }
    double[] params = stdpLearningRule.getParams();
    if (stdpLearningRule instanceof SymmetricSTPDLearningRule) {
      stdpLearningRule.setParams(IntStream.range(0, params.length).mapToDouble(i -> params[i] + random.nextGaussian() * Math.sqrt(SYMMETRIC_PARAMS_STD[i])).toArray());
    } else {
      stdpLearningRule.setParams(IntStream.range(0, params.length).mapToDouble(i -> params[i] + random.nextGaussian() * Math.sqrt(ASYMMETRIC_PARAMS_STD[i])).toArray());
    }
    return stdpLearningRule;
  }

}
