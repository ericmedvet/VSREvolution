package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.Crossover;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StandardEvolver;
import it.units.malelab.jgea.core.solver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.representation.tree.*;

import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class PairsTree implements NamedProvider<SolverBuilder<Tree<Pair<Double, Double>>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;
  private final double sigmaMut;
  private final int minTreeHeight;
  private final int maxTreeHeight;

  public PairsTree(double xOverProb, double tournamentRate, int minNTournament, double sigmaMut, int minTreeHeight, int maxTreeHeight) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
    this.sigmaMut = sigmaMut;
    this.minTreeHeight = minTreeHeight;
    this.maxTreeHeight = maxTreeHeight;
  }

  @Override
  public SolverBuilder<Tree<Pair<Double, Double>>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<Tree<Pair<Double, Double>>, S, Q>, TotalOrderQualityBasedProblem<S, Q>, S> build(PrototypedFunctionBuilder<Tree<Pair<Double, Double>>, S> builder, S target) {
        IndependentFactory<Pair<Double, Double>> doublePairsFactory = random -> Pair.of(random.nextDouble(), random.nextDouble() * 2 - 1);
        Factory<Tree<Pair<Double, Double>>> factory = new RampedHalfAndHalf<>(minTreeHeight, maxTreeHeight, d -> 4, doublePairsFactory, doublePairsFactory);
        Crossover<Tree<Pair<Double, Double>>> treeCrossover = new SubtreeCrossover<>(maxTreeHeight);
        Mutation<Tree<Pair<Double, Double>>> treeMutation = new SubtreeMutation<>(maxTreeHeight, new GrowTreeBuilder<>(d -> 4, doublePairsFactory, doublePairsFactory));
        Mutation<Tree<Pair<Double, Double>>> treeGaussianMutation = (pairTree, random) -> Tree.map(
            pairTree,
            p -> Pair.of(p.first() + sigmaMut * random.nextGaussian(), p.second() + sigmaMut * random.nextGaussian())
        );
        Map<GeneticOperator<Tree<Pair<Double, Double>>>, Double> geneticOperators = Map.of(
            treeMutation, (1d - xOverProb) / 2,
            treeGaussianMutation, (1d - xOverProb) / 2,
            treeCrossover, xOverProb
        );
        if (!diversity) {
          return new StandardEvolver<>(
              builder.buildFor(target),
              factory,
              nPop,
              StopConditions.nOfFitnessEvaluations(nEval),
              geneticOperators,
              new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
              new Last(),
              nPop,
              true,
              remap,
              (p, r) -> new POSetPopulationState<>()
          );
        }
        return new StandardWithEnforcedDiversityEvolver<>(
            builder.buildFor(target),
            factory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>(),
            100
        );
      }
    };
  }
}
