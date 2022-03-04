package it.units.erallab.builder.solver;

import com.google.common.collect.Range;
import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Factory;
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
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import it.units.malelab.jgea.representation.tree.*;

import java.util.List;
import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class TreeAndDoubles implements NamedProvider<SolverBuilder<Pair<Tree<Double>, List<Double>>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;
  private final double sigmaMut;
  private final int minTreeHeight;
  private final int maxTreeHeight;

  public TreeAndDoubles(double xOverProb, double tournamentRate, int minNTournament, double sigmaMut, int minTreeHeight, int maxTreeHeight) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
    this.sigmaMut = sigmaMut;
    this.minTreeHeight = minTreeHeight;
    this.maxTreeHeight = maxTreeHeight;
  }

  @Override
  public SolverBuilder<Pair<Tree<Double>, List<Double>>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<Pair<Tree<Double>, List<Double>>, S, Q>, TotalOrderQualityBasedProblem<S, Q>, S> build(PrototypedFunctionBuilder<Pair<Tree<Double>, List<Double>>, S> builder, S target) {
        Pair<Tree<Double>, List<Double>> sampleGenotype = builder.exampleFor(target);
        int listSize = sampleGenotype.second().size();
        Factory<Pair<Tree<Double>, List<Double>>> factory = Factory.pair(
            new RampedHalfAndHalf<>(minTreeHeight, maxTreeHeight, d -> 4, new UniformDoubleFactory(0d, 1d), new UniformDoubleFactory(0d, 1d)),
            new FixedLengthListFactory<>(listSize, new UniformDoubleFactory(-1d, 1d)));
        Crossover<Tree<Double>> treeCrossover = new SubtreeCrossover<>(maxTreeHeight);
        Map<GeneticOperator<Pair<Tree<Double>, List<Double>>>, Double> geneticOperators = Map.of(
            Mutation.pair(
                new SubtreeMutation<>(maxTreeHeight, new GrowTreeBuilder<>(d -> 4, new UniformDoubleFactory(0d, 1d), new UniformDoubleFactory(0d, 1d))),
                new GaussianMutation(sigmaMut)
            ), 1d - xOverProb,
            Crossover.pair(
                treeCrossover,
                new GeometricCrossover(Range.closed(-.5d, 1.5d))
            ).andThen(
                Mutation.pair(Mutation.copy(), new GaussianMutation(sigmaMut))
            ), xOverProb
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
