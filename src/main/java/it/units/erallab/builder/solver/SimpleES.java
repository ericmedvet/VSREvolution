package it.units.erallab.builder.solver;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.SimpleEvolutionaryStrategy;
import it.units.malelab.jgea.core.solver.StopConditions;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public class SimpleES implements NamedProvider<SolverBuilder<List<Double>>> {

  private final double sigma;
  private final double parentsRate;

  public SimpleES(double sigma, double parentsRate) {
    this.sigma = sigma;
    this.parentsRate = parentsRate;
  }

  @Override
  public SolverBuilder<List<Double>> build(Map<String, String> params) {
    int nPop = Integer.parseInt(params.get("nPop"));
    int nEval = Integer.parseInt(params.get("nEval"));
    boolean remap = Boolean.parseBoolean(params.get("remap"));
    return new SolverBuilder<>() {
      @Override
      public <S, Q> IterativeSolver<? extends POSetPopulationState<List<Double>, S, Q>,
          TotalOrderQualityBasedProblem<S, Q>, S> build(
          PrototypedFunctionBuilder<List<Double>, S> builder, S target
      ) {
        return new SimpleEvolutionaryStrategy<>(
            builder.buildFor(target),
            new FixedLengthListFactory<>(builder.exampleFor(target).size(), new UniformDoubleFactory(-1d, 1d)),
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            (int) Math.round(nPop * parentsRate),
            1, sigma, remap
        );
      }
    };
  }
}
