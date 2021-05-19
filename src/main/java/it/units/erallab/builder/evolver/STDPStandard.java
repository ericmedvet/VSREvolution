package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.representation.STDPLearningRuleFactory;
import it.units.erallab.builder.evolver.representation.STDPLearningRuleMutation;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.ProbabilisticMutation;
import it.units.malelab.jgea.representation.sequence.SameTwoPointsCrossover;

import java.util.List;
import java.util.Map;

public class STDPStandard implements EvolverBuilder<List<STDPLearningRule>> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  protected final boolean diversityEnforcement;

  public STDPStandard(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
  }

  @Override
  public <T, F> Evolver<List<STDPLearningRule>, T, F> build(PrototypedFunctionBuilder<List<STDPLearningRule>, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    IndependentFactory<List<STDPLearningRule>> factory = new FixedLengthListFactory<>(length, new STDPLearningRuleFactory());
    Map<GeneticOperator<List<STDPLearningRule>>, Double> operators = Map.of(
        new ProbabilisticMutation<>(0.2, factory, new STDPLearningRuleMutation(0.5, new STDPLearningRuleFactory())), 1 - xOverProb,
        new SameTwoPointsCrossover<>(factory).andThen(new ProbabilisticMutation<>(0.2, factory, new STDPLearningRuleMutation(0.5, new STDPLearningRuleFactory()))), xOverProb
    );
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          factory,
          comparator.comparing(Individual::getFitness),
          nPop,
          operators,
          new Tournament(nTournament),
          new Worst(),
          nPop,
          true,
          true
      );
    }
    return new StandardWithEnforcedDiversityEvolver<>(
        builder.buildFor(target),
        factory,
        comparator.comparing(Individual::getFitness),
        nPop,
        operators,
        new Tournament(nTournament),
        new Worst(),
        nPop,
        true,
        true,
        100
    );
  }

}
