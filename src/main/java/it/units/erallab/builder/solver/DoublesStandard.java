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
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class DoublesStandard implements NamedProvider<SolverBuilder<List<Double>>> {

  private final double xOverProb;
  private final double tournamentRate;
  private final int minNTournament;
  private final double sigmaMut;

  public DoublesStandard(double xOverProb, double tournamentRate, int minNTournament, double sigmaMut) {
    this.xOverProb = xOverProb;
    this.tournamentRate = tournamentRate;
    this.minNTournament = minNTournament;
    this.sigmaMut = sigmaMut;
  }

  @Override
  public SolverBuilder<List<Double>> build(String name, Map<String, String> params) {
    if (!name.equals("numGA")) {
      throw new IllegalArgumentException();
    }
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean diversity = Boolean.parseBoolean(params.get("diversity"));
    boolean remap = Boolean.parseBoolean(params.get("remap"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<List<Double>, S, Q>,
          TotalOrderQualityBasedProblem<S, Q>, S> build(
          PrototypedFunctionBuilder<List<Double>, S> builder, S target
      ) {
        IndependentFactory<List<Double>> doublesFactory = new FixedLengthListFactory<>(builder.exampleFor(target)
            .size(), new UniformDoubleFactory(-1d, 1d));
        Map<GeneticOperator<List<Double>>, Double> geneticOperators = Map.of(
            new GaussianMutation(sigmaMut), 1d - xOverProb,
            new UniformCrossover<>(doublesFactory).andThen(new GaussianMutation(sigmaMut)), xOverProb
        );
        if (!diversity) {
          return new StandardEvolver<>(
              builder.buildFor(target),
              doublesFactory,
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
              doublesFactory,
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
