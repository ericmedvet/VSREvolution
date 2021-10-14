package it.units.erallab.builder.evolver;

import com.google.common.collect.Range;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.operator.Crossover;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import it.units.malelab.jgea.representation.tree.*;

import java.util.List;
import java.util.Map;

/**
 * @author giorgia
 */
public class TreeAndDoubles implements EvolverBuilder<Pair<Tree<Double>, List<Double>>> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  protected final boolean diversityEnforcement;

  private final int minTreeHeight;
  private final int maxTreeHeight;

  public TreeAndDoubles(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement, int minTreeHeight, int maxTreeHeight) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
    this.minTreeHeight = minTreeHeight;
    this.maxTreeHeight = maxTreeHeight;
  }

  public TreeAndDoubles(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement) {
    this(nPop, nTournament, xOverProb, diversityEnforcement, 3, 6);
  }

  @Override
  public <T, F> Evolver<Pair<Tree<Double>, List<Double>>, T, F> build(PrototypedFunctionBuilder<Pair<Tree<Double>, List<Double>>, T> builder, T target, PartialComparator<F> comparator) {
    Pair<Tree<Double>, List<Double>> sampleGenotype = builder.exampleFor(target);
    int listSize = sampleGenotype.second().size();
    Factory<Pair<Tree<Double>, List<Double>>> factory = Factory.pair(
        new RampedHalfAndHalf<>(minTreeHeight, maxTreeHeight, d -> 4, new UniformDoubleFactory(0d, 1d), new UniformDoubleFactory(0d, 1d)),
        new FixedLengthListFactory<>(listSize, new UniformDoubleFactory(-1d, 1d)));
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          factory,
          comparator.comparing(Individual::getFitness),
          nPop,
          getGeneticOperators(),
          new Tournament(nTournament),
          new Last(),
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
        getGeneticOperators(),
        new Tournament(nTournament),
        new Last(),
        nPop,
        true,
        true,
        100
    );
  }

  private Map<GeneticOperator<Pair<Tree<Double>, List<Double>>>, Double> getGeneticOperators() {
    Crossover<Tree<Double>> treeCrossover = new SubtreeCrossover<>(maxTreeHeight);
    return Map.of(
        Mutation.pair(
            new SubtreeMutation<>(maxTreeHeight, new GrowTreeBuilder<>(d -> 4, new UniformDoubleFactory(0d, 1d), new UniformDoubleFactory(0d, 1d))),
            new GaussianMutation(.35d)
        ), 1d - xOverProb,
        Crossover.pair(
            treeCrossover,
            new GeometricCrossover(Range.closed(-.5d, 1.5d))
        ).andThen(
            Mutation.pair(Mutation.copy(), new GaussianMutation(.1d))
        ), xOverProb
    );
  }

}
