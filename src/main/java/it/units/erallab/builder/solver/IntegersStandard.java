package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StandardEvolver;
import it.units.malelab.jgea.core.solver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.ProbabilisticMutation;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;

import java.util.List;
import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class IntegersStandard implements NamedProvider<SolverBuilder<List<Integer>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;

  public IntegersStandard(double xOverProb, double tournamentRate, int minNTournament) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
  }

  @Override
  public SolverBuilder<List<Integer>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<List<Integer>, S, Q>,
          TotalOrderQualityBasedProblem<S, Q>, S> build(
          PrototypedFunctionBuilder<List<Integer>, S> builder, S target
      ) {
        int length = builder.exampleFor(target).size();
        int maxValue = builder.exampleFor(target).get(0);
        IndependentFactory<List<Integer>> integersFactory = new FixedLengthListFactory<>(length, random -> random.nextInt(maxValue));
        double pMut = Math.max(0.01, 1d / (double) length);
        Map<GeneticOperator<List<Integer>>, Double> geneticOperators = Map.of(
            new ProbabilisticMutation<>(pMut, integersFactory, (i, random) -> random.nextInt(maxValue)), 1d - xOverProb,
            new UniformCrossover<>(integersFactory), xOverProb
        );
        if (!diversity) {
          return new StandardEvolver<>(
              builder.buildFor(target),
              integersFactory,
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
        } else {
          return new StandardWithEnforcedDiversityEvolver<>(
              builder.buildFor(target),
              integersFactory,
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
      }
    };
  }
}
