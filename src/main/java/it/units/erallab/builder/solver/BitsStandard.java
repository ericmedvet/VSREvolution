package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.StandardEvolver;
import it.units.malelab.jgea.core.solver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.bit.BitFlipMutation;
import it.units.malelab.jgea.representation.sequence.bit.BitString;
import it.units.malelab.jgea.representation.sequence.bit.BitStringFactory;

import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class BitsStandard implements NamedProvider<SolverBuilder<BitString>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;
  private final double pMut;

  public BitsStandard(double xOverProb, double tournamentRate, int minNTournament, double pMut) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
    this.pMut = pMut;
  }

  @Override
  public SolverBuilder<BitString> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.getOrDefault("diversity", "false"));
    boolean remap = Boolean.parseBoolean(params.getOrDefault("remap", "false"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<BitString, S, Q>, TotalOrderQualityBasedProblem<S,
          Q>, S> build(
          PrototypedFunctionBuilder<BitString, S> builder, S target
      ) {
        BitStringFactory bitsFactory = new BitStringFactory(builder.exampleFor(target).size());
        Map<GeneticOperator<BitString>, Double> geneticOperators = Map.of(
            new BitFlipMutation(Math.max(pMut, 1.5d / (double) builder.exampleFor(target).size())), 1d - xOverProb,
            new UniformCrossover<>(bitsFactory).andThen(new BitFlipMutation(pMut)), xOverProb
        );
        if (!diversity) {
          return new StandardEvolver<>(
              builder.buildFor(target),
              bitsFactory,
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
              bitsFactory,
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
