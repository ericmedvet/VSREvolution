package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.IndependentFactory;
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
import it.units.malelab.jgea.representation.tree.*;

import java.util.Map;

/**
 * @author giorgia
 */
public class PairsTree implements EvolverBuilder<Tree<Pair<Double, Double>>> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  private final boolean diversityEnforcement;
  private final int minTreeHeight;
  private final int maxTreeHeight;
  private final boolean remap;

  public PairsTree(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement, int minTreeHeight, int maxTreeHeight, boolean remap) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
    this.minTreeHeight = minTreeHeight;
    this.maxTreeHeight = maxTreeHeight;
    this.remap = remap;
  }

  public PairsTree(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement, boolean remap) {
    this(nPop, nTournament, xOverProb, diversityEnforcement, 3, 6, remap);
  }

  @Override
  public <T, F> Evolver<Tree<Pair<Double, Double>>, T, F> build(PrototypedFunctionBuilder<Tree<Pair<Double, Double>>, T> builder, T target, PartialComparator<F> comparator) {
    IndependentFactory<Pair<Double, Double>> doublePairsFactory = random -> Pair.of(random.nextDouble(), random.nextDouble() * 2 - 1);
    Factory<Tree<Pair<Double, Double>>> factory = new RampedHalfAndHalf<>(minTreeHeight, maxTreeHeight, d -> 4, doublePairsFactory, doublePairsFactory);
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          factory,
          comparator.comparing(Evolver.Individual::fitness),
          nPop,
          getGeneticOperators(),
          new Tournament(nTournament),
          new Last(),
          nPop,
          true,
          remap
      );
    }
    return new StandardWithEnforcedDiversityEvolver<>(
        builder.buildFor(target),
        factory,
        comparator.comparing(Evolver.Individual::fitness),
        nPop,
        getGeneticOperators(),
        new Tournament(nTournament),
        new Last(),
        nPop,
        true,
        remap,
        100
    );
  }

  private Map<GeneticOperator<Tree<Pair<Double, Double>>>, Double> getGeneticOperators() {
    Crossover<Tree<Pair<Double, Double>>> treeCrossover = new SubtreeCrossover<>(maxTreeHeight);
    IndependentFactory<Pair<Double, Double>> doublePairsFactory = random -> Pair.of(random.nextDouble(), random.nextDouble() * 2 - 1);
    Mutation<Tree<Pair<Double, Double>>> treeMutation = new SubtreeMutation<>(maxTreeHeight, new GrowTreeBuilder<>(d -> 4, doublePairsFactory, doublePairsFactory));
    final double sigma = 0.35;
    Mutation<Tree<Pair<Double, Double>>> treeGaussianMutation = (pairTree, random) -> Tree.map(
        pairTree,
        p -> Pair.of(p.first() + sigma * random.nextGaussian(), p.second() + sigma * random.nextGaussian())
    );
    return Map.of(
        treeMutation, (1d - xOverProb) / 2,
        treeGaussianMutation, (1d - xOverProb) / 2,
        treeCrossover, xOverProb
    );
  }

}
